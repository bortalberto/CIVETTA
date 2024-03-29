import binascii
import numpy as np
import os
#import ROOT as R
import glob2
import pandas as pd
from tqdm import tqdm
import warnings
from sklearn.cluster import KMeans
import sys
import configparser
import pickle
from multiprocessing import Pool, cpu_count
import itertools

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.ini"))
try:
    data_folder = config["GLOBAL"].get("data_folder")
    mapping_file = config["GLOBAL"].get("mapping_file")
    calib_folder = config["GLOBAL"].get("calib_folder")


except KeyError as E:
    print(f"{E}Missing or partial configration file, restore it.")

if data_folder == "TER":
    try:
        data_folder = os.environ["TER_data"]
    except KeyError as E:
        print(f"{E} is not defined in your system variables")

def checkConsecutive(l):
    return sorted(l) == list(range(min(l), max(l)+1))

class decoder:
    """
    This class decode the dat file and produces a root file
    You need to specify GEMROC_ID
    """

    def __init__(self, GEMROC_ID=0, RUN=0, SUBRUN=0, downsamplig=1):
        # self.GEMROC_ID = int(GEMROC_ID) Make more sense to specify GEMROC and subrun while calling the function, since they are different every time
        self.RUN = int(RUN)
        self.downsampling = downsamplig
        # self.SUBRUN = int(SUBRUN)

    def __del__(self):
        pass

    def decode_file(self, input_, root=False):
        """
        Write a root file from a data file
        :param path:
        :return:
        """
        path = input_[0]
        subRunNo = input_[1]
        GEMROC = input_[2]
        pd_list = []

        statinfo = os.stat(path)
        packet_header = -1
        packet_tailer = -1
        packet_udp = -1
        l1count = -1

        l1count_new = []
        lschannel = []
        lstac = []
        lstcoarse = []
        lstcoarse_10b = []
        lsecoarse = []
        lstfine = []
        lsefine = []
        lstigerid = []
        lsl1ts_min_tcoarse = []
        lslasttigerframenum = []
        lscount_mismatch = []
        lsdelta_coarse = []
        l1timestamp = -1
        gemroc = -1
        l1framenum = -1
        trailer_tiger = -1

        pre_timestamp = 0
        pre_pretimestamp = 0

        tiger_framenum = -1
        prev_tiger_framenum = -1
        prev2_tiger_framenum = -1
        prev3_tiger_framenum = -1

        flag_swap1 = False
        flag_swap2 = False
        firstPacket = True

        firstData = False
        print_debug = False

        with open(path, 'rb') as f:
            for i in range(0, statinfo.st_size // 8):
                data = f.read(8)
                if sys.version_info[0] == 2:
                    hexdata = str(binascii.hexlify(data))
                else:
                    hexdata = str(binascii.hexlify(data), 'ascii')
                string = "{:064b}".format(int(hexdata, 16))
                inverted = []
                for i in range(8, 0, -1):
                    inverted.append(string[(i - 1) * 8:i * 8])
                string_inv = "".join(inverted)
                int_x = int(string_inv, 2)

                ##############################################################################################
                ##																							##
                ##								TRIGGER-MATCH DECODE										##
                ##																							##
                ##############################################################################################

                if (((int_x & 0xE000000000000000) >> 61) == 0x6):  ##packet header
                    packet_header = 1
                    LOCAL_L1_COUNT_31_6 = int_x >> 32 & 0x3FFFFFF
                    LOCAL_L1_COUNT_5_0 = int_x >> 24 & 0x3F
                    LOCAL_L1_COUNT = (LOCAL_L1_COUNT_31_6 << 6) + LOCAL_L1_COUNT_5_0
                    LOCAL_L1_TIMESTAMP = int_x & 0xFFFF
                    pre_pretimestamp = pre_timestamp
                    pre_timestamp = l1timestamp
                    l1count = LOCAL_L1_COUNT
                    l1timestamp = LOCAL_L1_TIMESTAMP

                    if firstData:
                        if print_debug:
                            print("WARNING: not able to record last tiger frame number of previous packet: no hits from TIGER 0-3 (L1_count={})!!!!!!!!!!!!!!!".format(LOCAL_L1_COUNT))

                    firstData = True  ## Header flags that next line will be first data word of the packet

                    # if len(lschannel) > 0:  ## Non dovrebbe manco succedere
                    #     print("ERROR")
                    #     lschannel = []
                    #     lstac = []
                    #     lstcoarse = []
                    #     lsecoarse = []
                    #     lstfine = []
                    #     lsefine = []
                    #     lstcoarse_10b = []
                    #     lstigerid = []
                    #     lsl1ts_min_tcoarse = []
                    #     lslasttigerframenum = []
                    #     lscount_mismatch = []
                    #     l1count_new = []
                    #     lsdelta_coarse = []

                if (((int_x & 0xC000000000000000) >> 62) == 0x0 and packet_header == 1 and packet_udp != 1):  ## DATA word
                    LOCAL_L1_TS_minus_TIGER_COARSE_TS = LOCAL_L1_TIMESTAMP - ((int_x >> 32) & 0xFFFF)
                    # print "enter DATA"
                    lstigerid.append((int_x >> 59) & 0x7)
                    lschannel.append((int_x >> 50) & 0x3F)
                    lstac.append((int_x >> 48) & 0x3)
                    lsecoarse.append((int_x >> 20) & 0x3FF)
                    lstfine.append((int_x >> 10) & 0x3FF)
                    lsefine.append(int_x & 0x3FF)
                    lslasttigerframenum.append((int_x >> 56) & 0x7)

                    temp_ecoarse = (int_x >> 20) & 0x3FF
                    lstcoarse_10b.append(((int_x >> 32) & 0x3FF))
                    temp_tcoarse = ((int_x >> 32) & 0x3FF)

                    tcoarse = (int_x >> 32) & 0xFFFF
                    ecoarse = (int_x >> 20) & 0x3FF
                    if (((int_x >> 20) & 0x3FF) - ((int_x >> 32) & 0x3FF)) > 0:
                        lsdelta_coarse.append(((int_x >> 20) & 0x3FF) - ((int_x >> 32) & 0x3FF))
                    else:
                        lsdelta_coarse.append(((int_x >> 20) & 0x3FF) - ((int_x >> 32) & 0x3FF) + 1024)

                    lstcoarse.append(tcoarse)

                    count_mismatch = 0

                    lsl1ts_min_tcoarse_to_append = LOCAL_L1_TIMESTAMP - tcoarse
                    l1count_new_to_append = l1count

                    if int_x == 0 and print_debug:
                        print("WARNING: DATA with all zeros (subRun = {}, L1_count = {})".format(subRunNo, LOCAL_L1_COUNT))
                    else:
                        tiger_framenum = (int_x >> 56) & 0x7
                        if firstData:
                            prev3_tiger_framenum = prev2_tiger_framenum
                            prev2_tiger_framenum = prev_tiger_framenum
                            if (int_x >> 59) & 0x7 < 4:  ## store 2 previous tiger frame number (only from TIGER 0-3)
                                prev_tiger_framenum = tiger_framenum
                            else:
                                if print_debug:
                                    print("WARNING: not able to record last tiger frame number of this packet: no hits from TIGER 0-3 (L1_count={})!!!!!!!!!!!!!!!".format(LOCAL_L1_COUNT))

                            firstData = False

                        ########################################
                        ##         PACKETS MATCHING           ##
                        ########################################

                        ## Start from alignment of previous packet
                        if ((int_x >> 59) & 0x7 > 3):
                            if flag_swap1:
                                temp_diff = pre_timestamp - tcoarse
                            elif flag_swap2:
                                temp_diff = pre_pretimestamp - tcoarse
                            else:
                                temp_diff = LOCAL_L1_TIMESTAMP - tcoarse
                        else:
                            temp_diff = lsl1ts_min_tcoarse_to_append  ## TIGER 0-3 always take the current l1ts

                        ## Find correct packet
                        ## performed only when lsl1ts_min_tcoarse is not inside the trigger window (roll-over taken into account)
                        if (not ((temp_diff > 1299 and temp_diff < 1567) or (temp_diff < -63960 and temp_diff > -64240))):

                            if firstPacket:  ## avoid packets correction for first packet
                                pass  ## wrong entries in first packet should be discarded since they cannot be corrected
                            else:
                                # print("Try SWAP 0")         					## try swap packets by 0
                                temp_diff = LOCAL_L1_TIMESTAMP - tcoarse
                                if ((temp_diff > 1299 and temp_diff < 1567) or (temp_diff < -63960 and temp_diff > -64240)):
                                    if (flag_swap1 == True or flag_swap2 == True) and print_debug:
                                        print("SWAP 0 activated (L1_count={})".format(LOCAL_L1_COUNT))
                                    flag_swap1 = False
                                    flag_swap2 = False
                                else:
                                    # print("Try SWAP 1")         					## try swap packets by 1
                                    temp_diff = pre_timestamp - tcoarse
                                    if ((temp_diff > 1299 and temp_diff < 1567) or (temp_diff < -63960 and temp_diff > -64240)):
                                        if flag_swap1 == False and print_debug:
                                            print("SWAP 1 activated (L1_count={})".format(LOCAL_L1_COUNT))
                                        flag_swap1 = True
                                        flag_swap2 = False
                                    else:
                                        # print("Try SWAP 2")       					## try swap packets by 2
                                        temp_diff = pre_pretimestamp - tcoarse
                                        if ((temp_diff > 1299 and temp_diff < 1567) or (temp_diff < -63960 and temp_diff > -64240)):
                                            if flag_swap2 == False and print_debug:
                                                print("SWAP 2 activated (L1_count={})".format(LOCAL_L1_COUNT))
                                            flag_swap1 = False
                                            flag_swap2 = True
                                        elif print_debug:
                                            print("WARNING: not able to correct packet (L1_count={}) !!!!!!!!!!!!!!!".format(LOCAL_L1_COUNT))

                        ## Apply packet correction to data of TIGER 4-7
                        if ((int_x >> 59) & 0x7 > 3):  ## correct packet for data of TIGER 4-7
                            if not (flag_swap1 or flag_swap2):  ## apply SWAP by 0 packet
                                lsl1ts_min_tcoarse_to_append = LOCAL_L1_TIMESTAMP - tcoarse  ## use l1ts of current packet
                                l1count_new_to_append = l1count  ## use l1count of current packet
                                count_mismatch = 0
                            elif flag_swap1:  ## apply SWAP by 1 packet
                                lsl1ts_min_tcoarse_to_append = pre_timestamp - tcoarse  ## use l1ts of previous packet
                                l1count_new_to_append = l1count - 1  ## use l1count of previous packet
                                count_mismatch = 1
                                if tiger_framenum != prev2_tiger_framenum:
                                    if print_debug:
                                        print("TIGER framecount not matched (SWAP1: L1_count={}: {} vs {}) !!!!!!!!!!!!!!!".format(LOCAL_L1_COUNT, tiger_framenum, prev2_tiger_framenum))
                            elif flag_swap2:  ## apply SWAP by 2 packets
                                lsl1ts_min_tcoarse_to_append = pre_pretimestamp - tcoarse  ## use l1ts of 2 previous packet
                                l1count_new_to_append = l1count - 2  ## use l1count of 2 previous packet
                                count_mismatch = 2
                                if tiger_framenum != prev3_tiger_framenum:
                                    if print_debug:
                                        print("TIGER framecount not matched (SWAP2: L1_count={}: {} vs {}) !!!!!!!!!!!!!!!".format(LOCAL_L1_COUNT, tiger_framenum, prev3_tiger_framenum))
                            else:
                                print("Swap ERROR: a problem occurred in swap logic (subRun={}, L1_count={}) !!!!!!!!!!!!!!!".format(subRunNo, LOCAL_L1_COUNT))

                        ## Correct counters roll-over
                        if (lsl1ts_min_tcoarse_to_append < 0):
                            if ((int_x >> 59) & 0x7 > 3):
                                if flag_swap1:
                                    lsl1ts_min_tcoarse_to_append = pre_timestamp - tcoarse + 2 ** 16
                                if flag_swap2:
                                    lsl1ts_min_tcoarse_to_append = pre_pretimestamp - tcoarse + 2 ** 16
                                if not (flag_swap1 or flag_swap2):
                                    lsl1ts_min_tcoarse_to_append = LOCAL_L1_TIMESTAMP - tcoarse + 2 ** 16
                            else:
                                lsl1ts_min_tcoarse_to_append = LOCAL_L1_TIMESTAMP - tcoarse + 2 ** 16

                    #####################################################################################################################
                    #####################################################################################################################
                    #####################################################################################################################

                    lsl1ts_min_tcoarse.append(lsl1ts_min_tcoarse_to_append)
                    l1count_new.append(l1count_new_to_append)

                    lscount_mismatch.append(count_mismatch)

                if (((int_x & 0xE000000000000000) >> 61) == 0x7):  ## TRAILER WORD --> sometimes is missing --> DO NOT USE
                    # print "enter trailer"
                    packet_tailer = 1
                    l1framenum = (int_x >> 37) & 0xFFFFFF
                    trailer_tiger = (int_x >> 27) & 0x7
                    gemroc = (int_x >> 32) & 0x1F

                if (((int_x & 0xF000000000000000) >> 60) == 0x4):  ## UDP WORD --> used to flag end of packet
                    # print "enter UDP"
                    if packet_tailer == 0:
                        if print_debug:
                            print("WARNING: missing trailer word (subRun = {}, L1 count = {})!!!!!!!!!!!!!!!".format(subRunNo, LOCAL_L1_COUNT))
                    packet_udp = 1
                # pre_udp_packet = udp_packet
                # udp_packet = (((int_x >> 32)&0xFFFFF) + ((int_x >> 0) & 0xFFFFFFF))
                if (packet_header == 1 and packet_udp == 1):  ## Fill ROOT file
                    l_channel = []
                    l_tac = []
                    l_tcoarse = []
                    l_ecoarse = []
                    l_tfine = []
                    l_efine = []
                    l_tcoarse_10b = []
                    l_tiger = []
                    l_l1ts_min_tcoarse = []
                    l_lasttigerframenum = []
                    l_delta_coarse = []
                    l_count_ori = []
                    l_count = []
                    l_timestamp = []
                    l_gemroc = []
                    l_runNo = []
                    l_subRunNo = []
                    l_layer = []
                    l_l1framenum = []
                    for x in range(len(lstac)):
                        l_channel.append(lschannel.pop())
                        l_tac.append(lstac.pop())
                        l_tcoarse.append(lstcoarse.pop())
                        l_ecoarse.append(lsecoarse.pop())
                        l_tfine.append(lstfine.pop())
                        l_efine.append(lsefine.pop())
                        l_tcoarse_10b.append(lstcoarse_10b.pop())
                        l_tiger.append(lstigerid.pop())
                        l_l1ts_min_tcoarse.append(lsl1ts_min_tcoarse.pop())
                        l_lasttigerframenum.append(lslasttigerframenum.pop())
                        l_delta_coarse.append(lsdelta_coarse.pop())
                        l_count_ori.append(l1count)
                        l_count.append(l1count_new.pop())
                        l_timestamp.append(l1timestamp)
                        l_gemroc.append(GEMROC)
                        l_runNo.append(self.RUN)
                        l_subRunNo.append(subRunNo)
                        l_l1framenum.append(l1framenum)
                        if (GEMROC < 4):
                            l_layer.append(1)
                        elif (GEMROC > 3):
                            l_layer.append(2)
                        if (GEMROC > 11):
                            l_layer.append(0)

                    dict4_pd = {'channel'         : l_channel, 'tac': l_tac, 'tcoarse': l_tcoarse, "ecoarse": l_ecoarse, "tfine": l_tfine, "efine": l_efine, "tcoarse_10b": l_tcoarse_10b, "tiger": l_tiger,
                                "l1ts_min_tcoarse": l_l1ts_min_tcoarse, "lasttigerframenum": l_lasttigerframenum, "delta_coarse": l_delta_coarse, "count_ori": l_count_ori, "count": l_count, "timestamp": l_timestamp,
                                "gemroc"          : l_gemroc, "runNo": l_runNo, "subRunNo": l_subRunNo, "l1_framenum": l_l1framenum}
                    packet_header = 0
                    packet_tailer = 0
                    packet_udp = 0
                    firstPacket = False
                    # print (l1count)
                    # print (self.downsampling)
                    if l1count % self.downsampling == 0:
                        pd_list.append(pd.DataFrame(dict4_pd))
        if len(pd_list) > 0:
            final_pd = pd.concat(pd_list, ignore_index=True)
            if len(final_pd > 0):
                if root:
                    pass  ## Inibit root decode
                    # import root_pandas
                    # filename=path.replace(".dat", ".root")
                    # filename=filename.replace("raw_dat", "raw_root")
                    # filename=filename.replace("/RUN_", "/")
                    # root_pandas.to_root(final_pd,filename,"tree")
                else:
                    filename = path.replace(".dat", "-zstd.feather")
                    filename = filename.replace("raw_dat", "raw_root")
                    filename = filename.replace("/RUN_", "/")
                    # final_pd.to_pickle(filename, compression="gzip")
                    final_pd.to_feather(filename, compression="zstd")

    def decode_file_header_trailer(self, input_):
        """
        Write a root file from a data file
        :param path:
        :return:
        """
        path = input_[0]
        subRunNo = input_[1]
        GEMROC = input_[2]
        header_pd_dict = {}
        trailer_pd_dict = {}
        UDP_pd_dict = {}

        statinfo = os.stat(path)

        header_pd_dict["gemroc"] = []
        header_pd_dict["subrun"] = []
        header_pd_dict["run"] = []
        header_pd_dict["l1_count"] = []
        header_pd_dict["l1_ts"] = []
        header_pd_dict["top_L1_chk_error"] = []
        header_pd_dict["header_misalignment_error"] = []
        header_pd_dict["FIFO_FULL_error"] = []

        trailer_pd_dict["gemroc"] = []
        trailer_pd_dict["subrun"] = []
        trailer_pd_dict["run"] = []
        trailer_pd_dict["l1_frame"] = []
        trailer_pd_dict["tiger_id"] = []
        trailer_pd_dict["count_trailer"] = []
        trailer_pd_dict["ch"] = []
        trailer_pd_dict["last_count_from_ch"] = []
        trailer_pd_dict["l1_count"] = []

        UDP_pd_dict["UDP_num"] = []
        UDP_pd_dict["daq_pll_unlocked"] = []
        UDP_pd_dict["global_rx_error"] = []
        UDP_pd_dict["XCVR_rx_alignment_error"] = []
        UDP_pd_dict["l1_count"] = []

        LOCAL_L1_COUNT=0
        with open(path, 'rb') as f:
            for i in range(0, statinfo.st_size // 8):
                data = f.read(8)
                if sys.version_info[0] == 2:
                    hexdata = str(binascii.hexlify(data))
                else:
                    hexdata = str(binascii.hexlify(data), 'ascii')
                string = "{:064b}".format(int(hexdata, 16))
                inverted = []
                for i in range(8, 0, -1):
                    inverted.append(string[(i - 1) * 8:i * 8])
                string_inv = "".join(inverted)
                int_x = int(string_inv, 2)

                ##############################################################################################
                ##																							##
                ##								TRIGGER-MATCH DECODE										##
                ##																							##
                ##############################################################################################

                if (((int_x & 0xE000000000000000) >> 61) == 0x6):  ##packet header
                    header_pd_dict["gemroc"].append( GEMROC)
                    header_pd_dict["subrun"].append( subRunNo)
                    header_pd_dict["run"].append( self.RUN)

                    LOCAL_L1_COUNT_31_6 = int_x >> 32 & 0x3FFFFFF
                    LOCAL_L1_COUNT_5_0 = int_x >> 24 & 0x3F
                    LOCAL_L1_COUNT = (LOCAL_L1_COUNT_31_6 << 6) + LOCAL_L1_COUNT_5_0
                    LOCAL_L1_TIMESTAMP = int_x & 0xFFFF
                    header_pd_dict["l1_count"].append( LOCAL_L1_COUNT)
                    header_pd_dict["l1_ts"].append( LOCAL_L1_TIMESTAMP)

                    header_pd_dict["top_L1_chk_error"].append( (int_x >> 58)& 0x1)
                    header_pd_dict["header_misalignment_error"].append( (int_x >> 59)& 0x1)
                    header_pd_dict["FIFO_FULL_error"].append( (int_x >> 60)& 0x1)



                if (((int_x & 0xC000000000000000) >> 62) == 0x0):  ## DATA word
                    pass

                if (((int_x & 0xE000000000000000) >> 61) == 0x7):  ## TRAILER WORD --> sometimes is missing --> DO NOT USE
                    # print "enter trailer"

                    trailer_pd_dict["gemroc"].append(GEMROC)
                    trailer_pd_dict["subrun"].append(subRunNo)
                    trailer_pd_dict["run"].append(self.RUN)
                    trailer_pd_dict["l1_count"].append(LOCAL_L1_COUNT)
                    trailer_pd_dict["l1_frame"].append(((int_x >> 37) & 0xFFFFFF))
                    trailer_pd_dict["tiger_id"].append(((int_x >> 27) & 0x7))
                    trailer_pd_dict["count_trailer"].append(((int_x >> 24) & 0x7))
                    trailer_pd_dict["ch"].append(((int_x >> 18) & 0x3F))
                    trailer_pd_dict["last_count_from_ch"].append((int_x & 0x3FFFF))

                if (((int_x & 0xF000000000000000) >> 60) == 0x4):  ## UDP WORD --> used to flag end of packet
                    UDP_pd_dict["UDP_num"].append(((int_x >> 32) & 0xFFFFF) + ((int_x >> 0) & 0xFFFFFFF))
                    UDP_pd_dict["daq_pll_unlocked"].append((int_x >> 57)& 0x1)
                    UDP_pd_dict["global_rx_error"].append((int_x >> 58)& 0x1)
                    UDP_pd_dict["XCVR_rx_alignment_error"].append((int_x >> 59)& 0x1)
                    UDP_pd_dict["l1_count"].append(LOCAL_L1_COUNT)


        # print (header_pd_dict)
        if len(header_pd_dict) > 0 and len(trailer_pd_dict) > 0 and len(UDP_pd_dict) > 0:
            header_pd = pd.DataFrame(header_pd_dict)
            trailer_pd = pd.DataFrame(trailer_pd_dict)
            UDP_pd = pd.DataFrame(UDP_pd_dict)
            return header_pd, trailer_pd, UDP_pd
        else:
            return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()

class calib:
    """
    Class created to apply calibration and mapping
    """

    def __init__(self, run_number, calib_folder, mapping_file, data_folder, root_dec, cylinder, cosmic = False):
        self.run_number = run_number
        self.calib_folder = calib_folder
        self.mapping_file = mapping_file
        self.data_folder = data_folder
        self.root_dec = False
        self.cylinder = cylinder
        self.cosmic = cosmic

    def load_mapping(self):
        """
        Loads the mapping file
        :return:
        """
        if self.mapping_file[-5:] == ".root":
            self.load_mapping_root()
        else:
            try:
                mapping_pd = pd.read_pickle(self.mapping_file)
                self.mapping_pd = mapping_pd
            except Exception as E:
                print(f"Can't load the mapping file, exception: {E}. Verify that the file ({self.mapping_file})exists and it's readable")
                sys.exit(1)

    def load_mapping_root(self):
        """
        Loads the mapping file
        :return:
        """
        try:
            import uproot
            file = uproot.open(self.mapping_file)
            mapping_pd = file["tree"].arrays(library="pd")
            self.mapping_pd = mapping_pd
            self.mapping_pd["tiger"] = self.mapping_pd["SW_FEB_id"]
            self.mapping_pd["strip_x"] = self.mapping_pd["pos_x"]
            self.mapping_pd["strip_y"] = self.mapping_pd["pos_v"]
            self.mapping_pd["planar"] = self.mapping_pd["layer_id"]
            self.mapping_pd["HW_feb_id"] = self.mapping_pd["HW_FEB_id"]


        except Exception as E:
            print(f"Can't load the mapping file, exception: {E}. Verify that the file ({self.mapping_file})exists and it's readable")
            sys.exit(1)




    def get_channels_QDC_calib(self, HW_FEB, layer):
        """
        The first time we try to access a calibration, we load the txt file, in such way we load only the needed files
        :param HW_FEB:
        :param layer:
        :return:
        """
        fname_1 = "{2}/QDC/L{0}_QDC_calib/L{0}FEB{1}_c1_Efine_calib.txt".format(layer, HW_FEB, self.calib_folder)
        fname_2 = "{2}/QDC/L{0}_QDC_calib/L{0}FEB{1}_c2_Efine_calib.txt".format(layer, HW_FEB, self.calib_folder)
        if not os.path.isfile(fname_1):
            print(f"Can't find {fname_1}")
            exit(0)
        if not os.path.isfile(fname_2):
            print(f"Can't find {fname_2}")
            exit(0)
        return {
            0: np.genfromtxt(fname=fname_1, converters={0: convert_none, 1: convert_none}),
            1: np.genfromtxt(fname=fname_2, converters={0: convert_none, 1: convert_none}),
        }

    def get_channels_TAC_calib(self, HW_FEB, layer):
        """
        The first time we try to access a calibration, we load the txt file, in such way we load only the needed files
         :param HW_FEB:
         :param layer:
         :return:
         """
        return {
            0: np.loadtxt(fname="{2}/L{0}_TDC/L{0}FEB{1}_c1_Efine_calib.txt".format(layer, HW_FEB, calib_folder)),
            1: np.loadtxt(fname="{2}/L{0}_TDC/L{0}FEB{1}_c2_Efine_calib.txt".format(layer, HW_FEB, calib_folder)),
        }

    def build_mapping_group(self):
        self.mapping_group = self.mapping_pd.groupby(["channel_id", "tiger", "gemroc_id"])

    def get_mapping_value(self, field_names, channel_id, tiger, gemroc):
        """
        Return the mapped value for the a certain channel
        :param field_names:
        :param channel_id:
        :param tiger:
        :param gemroc:
        :return:
        """
        return_list = []
        # mapping_pd=self.mapping_pd[(self.mapping_pd.channel_id == channel_id) & (self.mapping_pd.tiger == tiger) & (self.mapping_pd.gemroc_id == gemroc)]
        mapping_pd = self.mapping_group.get_group((channel_id, tiger, gemroc))
        for field_name in field_names:
            return_list.append(int(mapping_pd.iloc[0][field_name]))
        return return_list

    def calibrate_charge(self, calib_dict, HW_feb_id, planar, tiger, channel, efine):
        """
        Calculate the charge, given efine and the channel
        :param calib_dict:
        :param HW_feb_id:
        :param tiger:
        :param channel:
        :param efine:
        :return:
        """
        if self.cylinder:
            constant = calib_dict[HW_feb_id, planar][int(tiger % 2)][channel][1]
            slope = calib_dict[HW_feb_id, planar][int(tiger % 2)][channel][2]
        else:
            constant = calib_dict[HW_feb_id, 3][int(tiger % 2)][channel][1]
            slope = calib_dict[HW_feb_id, 3][int(tiger % 2)][channel][2]

        if slope==0:
            return np.nan
        if (efine >= 1008):
            charge_SH = (((-1 * constant) - (1024 - efine)) / slope)
        else:
            charge_SH = ((-1 * constant + efine) / slope)

        if charge_SH < -200 or charge_SH > 200 or charge_SH==np.inf or charge_SH==-np.inf:
            print (f"Warning, strange charge value, check calibration layer {planar}, HW_feb_id{HW_feb_id}, channel {channel}")
            charge_SH = 0
        return charge_SH


    # def calibrate_subrun(self,subrun):
    # """
    # usit for profiling
    # """
    #     import cProfile
    #     subrun=int(subrun)
    #     cProfile.runctx('self.calibrate_subrun_runner(subrun)', globals(), locals(), 'prof%d.prof' %subrun)

    def calibrate_subrun(self, subrun):
        """
        Calirate a single subrun
        :param subrun:
        :return:
        """
        decode_pd = pd.read_feather("{}/raw_root/{}/Sub_RUN_dec_{}-zstd.feather".format(self.data_folder, self.run_number, subrun))
        ana_pd = decode_pd
        self.build_mapping_group()

        ana_pd["data"] = [self.get_mapping_value(("strip_x", "strip_y", "planar", "FEB_label", "HW_feb_id"), *a) for a in tuple(zip(ana_pd["channel"], ana_pd["tiger"], ana_pd["gemroc"]))]
        ana_pd["subrun"] = subrun
        ana_pd["hit_id"] = ana_pd.index
        ana_pd[["strip_x", "strip_y", "planar", "FEB_label", "HW_feb_id"]] = pd.DataFrame(ana_pd.data.tolist())
        ana_pd = ana_pd.drop(columns=["data"])
        ana_pd = ana_pd.astype(int)
        calib_dict = {}
        if self.cylinder:
            for HW_feb_id, planar in ana_pd.groupby(["HW_feb_id", "planar"]).groups.keys():
                calib_dict[(int(HW_feb_id), int(planar))] = self.get_channels_QDC_calib(int(HW_feb_id), int(planar))
        else:
            for HW_feb_id in ana_pd["HW_feb_id"].unique():
                calib_dict[(int(HW_feb_id), 3)] = self.get_channels_QDC_calib(int(HW_feb_id), 3)

        ana_pd["charge_SH"] = [self.calibrate_charge(calib_dict, *a) for a in tuple(zip(ana_pd["HW_feb_id"], ana_pd["planar"], ana_pd["tiger"], ana_pd["channel"], ana_pd["efine"]))]
        compress_pd = compress_hit_pd(ana_pd)
        verifiy_compression_validity(ana_pd, compress_pd)
        ana_pd = compress_pd
        if self.cylinder:
            ana_pd = self.create_cyl_names(ana_pd)
        # import root_pandas
        # root_pandas.to_root(ana_pd,"{}/raw_root/{}/Sub_RUN_pl_ana_{}.root".format(self.data_folder,self.run_number,subrun),"tree")
        # ana_pd.to_pickle("{}/raw_root/{}/Sub_RUN_pl_ana{}.pickle.gzip".format(self.data_folder, self.run_number, subrun), compression="gzip")
        ana_pd.to_feather("{}/raw_root/{}/Sub_RUN_pl_ana{}-zstd.feather".format(self.data_folder, self.run_number, subrun))
        return ana_pd

    def create_cyl_names(self, run_data):
        run_data["view"] = ""
        run_data["strip"] = np.nan
        run_data.loc[run_data.strip_x > -1, "view"] = "X"
        run_data.loc[run_data.strip_y > -1, "view"] = "V"
        run_data.loc[run_data.view == "X", "strip"] = run_data.loc[run_data.view == "X", "strip_x"]
        run_data.loc[run_data.view == "V", "strip"] = run_data.loc[run_data.view == "V", "strip_y"]
        return run_data

    def create_hits_pd_and_single_root(self):
        """
        Convert the root ana files in a dataframe and store it in pickle compressed format.
        :return:
        """

        data_pd = pd.DataFrame()
        if self.root_dec:
            pass
            # import root_pandas
            #
            # for filename in glob2.iglob("{}/raw_root/{}/Sub_RUN_pl_ana*.root".format(self.data_folder, self.run_number)):
            #     f = R.TFile.Open(filename)
            #     if f.tree.GetEvent() > 0:
            #         data_pd = data_pd.append(root_pandas.read_root(filename, "tree"), ignore_index=True)
            # data_pd.to_pickle("{}/raw_root/{}/hit_data.pickle.gzip".format(self.data_folder, self.run_number), compression="gzip")
            # root_pandas.to_root(data_pd, "{}/raw_root/{}/pl_ana.root".format(self.data_folder, self.run_number), "tree")


        else:
            pd_list = []
            for filename in glob2.iglob("{}/raw_root/{}/Sub_RUN_pl_ana*-zstd.feather".format(self.data_folder, self.run_number)):
                pd_list.append(pd.read_feather(filename))
            data_pd = pd.concat(pd_list, ignore_index=True)
            # data_pd.to_pickle("{}/raw_root/{}/hit_data.pickle.gzip".format(self.data_folder, self.run_number), compression="gzip")
            if self.cosmic:
                data_pd["count"] = data_pd.groupby(["subRunNo", "count"]).ngroup()
            data_pd.to_feather("{}/raw_root/{}/hit_data-zstd.feather".format(self.data_folder, self.run_number))

    def append_hits_pd_and_single_root(self):
        """
        Same as above, but appends if data exists
        :return:
        """
        if self.root_dec:
            pass
            # import root_pandas
            # path = self.data_folder + f"/raw_root/{self.run_number}/hit_data.pickle.gzip"
            #
            # if os.path.isfile(path):
            #     data_pd=pd.read_pickle(path, compression="gzip")
            # else:
            #     data_pd=pd.DataFrame()
            # for filename in glob2.iglob("{}/raw_root/{}/Sub_RUN_pl_ana*.root".format(self.data_folder, self.run_number)):
            #     f = R.TFile.Open(filename)
            #     if f.tree.GetEvent() > 0:
            #         data_pd = data_pd.append(root_pandas.read_root(filename, "tree"))
            #
            # data_pd.to_pickle("{}/raw_root/{}/hit_data.pickle.gzip".format(self.data_folder, self.run_number), compression="gzip")
            # root_pandas.to_root(data_pd, "{}/raw_root/{}/pl_ana.root".format(self.data_folder, self.run_number), "tree")
        else:
            path = self.data_folder + f"/raw_root/{self.run_number}/hit_data-zstd.feather"
            if os.path.isfile(path):
                data_pd = pd.read_feather(path)
            else:
                data_pd = pd.DataFrame()

            pd_list = []
            for filename in glob2.iglob("{}/raw_root/{}/Sub_RUN_pl_ana*-zstd.feather".format(self.data_folder, self.run_number)):
                pd_list.append(pd.read_feather(filename))
            data_pd = pd.concat(pd_list, ignore_index=True)
            data_pd.to_feather("{}/raw_root/{}/hit_data-zstd.feather".format(self.data_folder, self.run_number))

            # data_pd.to_pickle("{}/raw_root/{}/hit_data.pickle.gzip".format(self.data_folder, self.run_number), compression="gzip")
    #
    # mapping_pd=load_mapping()
    #
    #
    # file_list=[]
    # for filename, subrun in glob2.iglob("{}/{}/Sub_RUN_dec_*.root".format(data_folder,run_number),with_matches=True):
    #     file_list.append(filename)
    # for filename, subrun in tqdm(glob2.iglob("{}/{}/Sub_RUN_dec_*.root".format(data_folder,run_number),with_matches=True),total=len(file_list)):
    #     (calibrate_subrun(subrun[0]))
    #


class clusterize:
    """
    Class created to clusterize the data, in both 1-D and 2-D
    """

    def __init__(self, run_number, data_folder, signal_window_lower_limit, signal_window_upper_limit):
        """
        It needs the pickle file to run
        :param run_number:
        :param data_folder:
        """
        self.run_number = run_number
        self.data_folder = data_folder
        self.signal_window_lower_limit = signal_window_lower_limit
        self.signal_window_upper_limit = signal_window_upper_limit

    @classmethod
    def default_time_winw(cls, run_number, data_folder):
        config = configparser.ConfigParser()
        config.read(os.path.join(sys.path[0], "config.ini"))
        signal_window_lower_limit_conf = config["GLOBAL"].get("signal_window_lower_limit")
        signal_window_upper_limit_conf = config["GLOBAL"].get("signal_window_upper_limit")
        return cls(run_number, data_folder, signal_window_lower_limit_conf, signal_window_upper_limit_conf)

    def load_data_pd(self, subrunNo_tgt=None):
        """
        Load the pickle file with the single hit information
        :return:
        """
        self.data_pd = pd.read_feather("{}/raw_root/{}/hit_data-zstd.feather".format(self.data_folder, self.run_number))
        if subrunNo_tgt:
            data_pd_cut_0 = self.data_pd[(self.data_pd.runNo == self.run_number) & (self.data_pd.l1ts_min_tcoarse > int(self.signal_window_lower_limit)) & (self.data_pd.l1ts_min_tcoarse < int(self.signal_window_upper_limit)) & (self.data_pd.delta_coarse > 0)]
            data_pd_cut_0 = data_pd_cut_0[data_pd_cut_0.subRunNo == subrunNo_tgt]
            self.data_pd = data_pd_cut_0

    def read_subruns(self):
        """
        Returns the list of subruns in the run
        :return:
        """
        return (self.data_pd.subRunNo.unique())



    def clusterize_view_old(self, data_pd, view):
        """
        Recursively search for the number of cluster which minimize the distance. Tolerates holes in the cluster
        :param hit_pos:
        :param hit_charge:
        :return:
        """
        hit_pos = data_pd[f"strip_{view}"].to_numpy()
        hit_charge = data_pd.charge_SH.to_numpy()
        hit_id = data_pd.hit_id.to_numpy()

        k = 1  # Initialize with 1 cluster

        while True:
            ret_clusters = []
            KM = KMeans(n_clusters=k, n_init=1)  # Initialize the algorithm with k clusters
            KM.fit(hit_pos.reshape(-1, 1))  # Perform the alg and find the clusters centers

            for n, c in enumerate(KM.cluster_centers_):  # For each cluster
                hit_pos_this_c = hit_pos[KM.labels_ == n]  # Load hit position and hit charge for this cluster
                included = (abs(hit_pos_this_c - c) < len(hit_pos_this_c) / 2 + 1)  # For each hit, checks if the hit is in cluster_size/2 +1 from the center
                if np.any(included != True):
                    k += 1
                else:
                    hit_charge_this_c = hit_charge[KM.labels_ == n]
                    hit_id_this_c = hit_id[KM.labels_ == n]
                    ret_clusters.append((charge_centroid(hit_pos_this_c, hit_charge_this_c), np.sum(hit_charge_this_c), len(hit_pos_this_c), hit_id_this_c))  # pos,charge, size

            if len(ret_clusters) == len(KM.cluster_centers_):
                return ret_clusters

    def clusterize_view(self, data_pd, view):
        """
        Recursively search for the number of cluster which minimize the distance. Tolerates holes in the cluster
        :param hit_pos:
        :param hit_charge:
        :return:
        """
        hit_pos = data_pd[f"strip_{view}"].to_numpy()
        hit_charge = data_pd.charge_SH.to_numpy()
        hit_id = data_pd.hit_id.to_numpy()

        k = 1  # Initialize with 1 cluster
        centers = np.random.uniform(0, 128, k)

        while True:
            #     print (k)
            ret_clusters = []
            if len(hit_pos) == 0:
                print("No hit")
                break
            cluster_centers, labels = manual_kmean(hit_pos, centers)  # Initialize the algorithm with k clusters
            #     print (cluster_centers)

            good_clusters = 0
            for n, c in enumerate(cluster_centers):  # For each cluster
                hit_pos_this_c = hit_pos[labels == n]  # Load hit position and hit charge for this cluster
                distance = abs(hit_pos_this_c - c)  # Distance from center
                included = (abs(hit_pos_this_c - c) < len(hit_pos_this_c) / 2 + 1)  # For each hit, checks if the hit is in cluster_size/2 +1 from the center
                #         print (included)
                if np.any(included != True):
                    k += 1
                    centers = np.append(cluster_centers, hit_pos_this_c[np.argmax(distance)])  # Add to the center list the farthest point
                    break
                else:
                    good_clusters = good_clusters + 1
                    # hit_charge_this_c = hit_charge[labels == n]
                    # hit_id_this_c = hit_id[labels == n]
                    # ret_clusters.append( (self.charge_centroid(hit_pos_this_c, hit_charge_this_c), np.sum(hit_charge_this_c), len(hit_pos_this_c ),hit_id_this_c ) )  # pos,charge, size

            labels_list = [x for _, x in sorted(zip(cluster_centers, set(labels)))]  # Order labels by center
            if len(cluster_centers) == good_clusters:
                # Merging near clusters:
                if len(set(labels)) > 1:
                    i = 0
                    running = True
                    while running:
                        n, m = labels_list[i], labels_list[i + 1]
                        gr_1 = (hit_pos[labels == n])
                        gr_2 = (hit_pos[labels == m])
                        if (abs(max(gr_1) - min(gr_2)) < 3) or (abs(min(gr_1) - max(gr_2)) < 3):
                            labels[labels == n] = m
                            cluster_centers = []
                            for label in set(labels):
                                hit_pos_this_c = hit_pos[labels == label]
                                cluster_centers.append(np.mean(hit_pos_this_c))
                            labels_list = [x for _, x in sorted(zip(cluster_centers, set(labels)))]
                            i = 0
                        else:
                            i = i + 1
                        if i >= len(labels_list) - 1:
                            break
                cluster_centers = []
                # End merging near clusters

                for label in set(labels):
                    hit_pos_this_c = hit_pos[labels == label]
                    cluster_centers.append(np.mean(hit_pos_this_c))
                    hit_charge_this_c = hit_charge[labels == label]
                    hit_id_this_c = hit_id[labels == label]
                    consecutive =  checkConsecutive(hit_pos_this_c)
                    ret_clusters.append((charge_centroid(hit_pos_this_c, hit_charge_this_c), np.sum(hit_charge_this_c), len(hit_pos_this_c), hit_id_this_c, consecutive))  # pos,charge, size
                return ret_clusters

    def build_view_clusters(self, data_pd):
        """
        Builds the cluster for one run (the run selection part is redundant). The subrunNo param allows parallelization
        :param subrunNo:
        :return:
        """

        dict_4_pd = {
            "run"      : [],
            "subrun"   : [],
            "count"    : [],
            "planar"   : [],
            "cl_pos_x" : [],
            "cl_pos_y" : [],
            "cl_charge": [],
            "cl_size"  : [],
            "consecutive" : [],
            "cl_id"    : [],
            "hit_ids"  : []

        }
        for runNo in data_pd["runNo"].unique():
            data_pd_cut_1 = data_pd[(data_pd.runNo == runNo) & (data_pd.l1ts_min_tcoarse > int(self.signal_window_lower_limit)) & (data_pd.l1ts_min_tcoarse < int(self.signal_window_upper_limit)) & (data_pd.delta_coarse > 0)]
            for count in data_pd_cut_1["count"].unique():
                data_pd_cut_2 = data_pd_cut_1[data_pd_cut_1["count"] == count]
                for planar in data_pd_cut_2["planar"].unique():
                    data_pd_cut_3 = data_pd_cut_2[data_pd_cut_2.planar == planar]
                    for view in ("x", "y"):
                        clusters = []
                        data_pd_cut_4 = data_pd_cut_3[data_pd_cut_3[f"strip_{view}"] > -1]
                        if len(data_pd_cut_4) > 0:
                            clusters = self.clusterize_view(data_pd_cut_4, view)
                        for n, cluster in enumerate(clusters):
                            dict_4_pd["run"].append(runNo)
                            dict_4_pd["subrun"].append(int(data_pd.subRunNo.mean()))
                            dict_4_pd["count"].append(count)
                            dict_4_pd["planar"].append(planar)
                            if view == "x":
                                dict_4_pd["cl_pos_x"].append(cluster[0])
                                dict_4_pd["cl_pos_y"].append(np.nan)
                            else:
                                dict_4_pd["cl_pos_y"].append(cluster[0])
                                dict_4_pd["cl_pos_x"].append(np.nan)
                            dict_4_pd["cl_charge"].append(cluster[1])
                            dict_4_pd["cl_size"].append(cluster[2])
                            dict_4_pd["hit_ids"].append(cluster[3])
                            dict_4_pd["consecutive"].append(bool(cluster[4]))
                            dict_4_pd["cl_id"].append(int(n))

        return (pd.DataFrame(dict_4_pd))

    def save_cluster_pd(self, subrun="All"):
        """
        Updating to feather format
        """
        # if subrun == "All":
        #     self.cluster_pd.to_pickle("{}/raw_root/{}/cluster_pd_1D.pickle.gzip".format(self.data_folder, self.run_number), compression="gzip")
        # else:
        #     self.cluster_pd.to_pickle("{}/raw_root/{}/cluster_pd_1D_sub_{}.pickle.gzip".format(self.data_folder, self.run_number, subrun), compression="gzip")
        if subrun == "All":
            self.cluster_pd.reset_index(drop=True, inplace=True)
            self.cluster_pd.to_feather("{}/raw_root/{}/cluster_pd_1D-zstd.feather".format(self.data_folder, self.run_number), compression="zstd")
        else:
            self.cluster_pd.reset_index(drop=True, inplace=True)
            self.cluster_pd.to_feather("{}/raw_root/{}/cluster_pd_1D_sub_{}-zstd.feather".format(self.data_folder, self.run_number, subrun), compression="zstd")


    def append_cluster_pd(self):
        path = "{}/raw_root/{}/cluster_pd_1D-zstd.feather".format(self.data_folder, self.run_number)
        if os.path.isfile(path):
            cluster_pd_old = pd.read_feather(path)
            self.cluster_pd = pd.concat((self.cluster_pd, cluster_pd_old))
        self.cluster_pd.reset_index(drop=True, inplace=True)
        self.cluster_pd.to_feather("{}/raw_root/{}/cluster_pd_1D-zstd.feather".format(self.data_folder, self.run_number), compression="zstd")

    def load_cluster_pd(self, subrun="All"):
        try :
            if subrun == "All":
                self.cluster_pd = pd.read_feather("{}/raw_root/{}/cluster_pd_1D-zstd.feather".format(self.data_folder, self.run_number))
            else:
                self.cluster_pd = pd.read_feather("{}/raw_root/{}/cluster_pd_1D_sub_{}-zstd.feather".format(self.data_folder, self.run_number, subrun))
        except:
            if subrun == "All":
                self.cluster_pd = pd.read_pickle("{}/raw_root/{}/cluster_pd_1D.pickle.gzip".format(self.data_folder, self.run_number), compression="gzip")
            else:
                self.cluster_pd = pd.read_pickle("{}/raw_root/{}/cluster_pd_1D_sub_{}.pickle.gzip".format(self.data_folder, self.run_number, subrun), compression="gzip")

    def build_2D_clusters(self, cluster_pd):
        dict_4_pd = {
            "run"        : [],
            "subrun"     : [],
            "count"      : [],
            "planar"     : [],
            "cl_pos_x"   : [],
            "cl_pos_y"   : [],
            "cl_charge"  : [],
            "cl_charge_x": [],
            "cl_charge_y": [],
            "cl_size_x"  : [],
            "cl_size_y"  : [],
            "cl_size_tot": []
        }
        events_pd_clusters = cluster_pd.groupby(["count", "planar"])
        for key in events_pd_clusters.groups:
            event_pd_cl = events_pd_clusters.get_group(key)
            cls_x = event_pd_cl[event_pd_cl.cl_pos_x.notna()]
            cls_y = event_pd_cl[event_pd_cl.cl_pos_y.notna()]
            if (cls_x.shape[0] > 0) and (cls_y.shape[0] > 0):
                cl_x = event_pd_cl.loc[cls_x.cl_charge.idxmax(axis=0)]
                cl_y = event_pd_cl.loc[cls_y.cl_charge.idxmax(axis=0)]
                dict_4_pd["run"].append(self.run_number)
                dict_4_pd["subrun"].append(cl_x.subrun)
                dict_4_pd["count"].append(key[0])
                dict_4_pd["planar"].append(key[1])
                dict_4_pd["cl_pos_x"].append(cl_x.cl_pos_x)
                dict_4_pd["cl_pos_y"].append(cl_y.cl_pos_y)
                dict_4_pd["cl_charge"].append(cl_x.cl_charge + cl_y.cl_charge)
                dict_4_pd["cl_charge_x"].append(cl_x.cl_charge)
                dict_4_pd["cl_charge_y"].append(cl_y.cl_charge)
                dict_4_pd["cl_size_x"].append(cl_x.cl_size)
                dict_4_pd["cl_size_y"].append(cl_y.cl_size)
                dict_4_pd["cl_size_tot"].append(cl_x.cl_size + cl_y.cl_size)
        return (pd.DataFrame(dict_4_pd))

    def save_cluster_pd_2D(self, subrun="All"):
        if subrun == "All":
            self.cluster_pd_2D.reset_index(drop=True, inplace=True)
            self.cluster_pd_2D.to_feather("{}/raw_root/{}/cluster_pd_2D-zstd.feather".format(self.data_folder, self.run_number), compression="zstd")
        else:
            self.cluster_pd_2D.to_feather("{}/raw_root/{}/cluster_pd_2D_sub_{}-zstd.feather".format(self.data_folder, self.run_number, subrun), compression="zstd")

    def append_cluster_pd_2D(self):
        """
        Append the data from the last subrun with the others
        :return:
        """
        path = "{}/raw_root/{}/cluster_pd_2D-zstd.feather".format(self.data_folder, self.run_number)
        if os.path.isfile(path):
            cluster_pd_2D_old = pd.read_feather(path)
            self.cluster_pd_2D = pd.concat((self.cluster_pd_2D, cluster_pd_2D_old))
        self.cluster_pd_2D.reset_index(drop=True, inplace=True)
        self.cluster_pd_2D.to_feather("{}/raw_root/{}/cluster_pd_2D-zstd.feather".format(self.data_folder, self.run_number), compression="zstd")


class tracking_2d:
    """
    Simple tracking 2D 4 data selection
    """

    def __init__(self, run_number, data_folder):
        """
        It needs the cluster 2_D pickle file to run
        :param run_number:
        :param data_folder:
        """
        self.run_number = run_number
        self.data_folder = data_folder

    def load_cluster_2D(self):
        """
        Load the cluster 2-D file
        :return:
        """
        cluster_pd_2D = pd.read_feather("{}/raw_root/{}/cluster_pd_2D-zstd.feather".format(self.data_folder, self.run_number))
        cluster_pd_2D["cl_pos_x_cm"] = cluster_pd_2D.cl_pos_x * 0.0650
        cluster_pd_2D["cl_pos_y_cm"] = cluster_pd_2D.cl_pos_y * 0.0650
        cluster_pd_2D["cl_pos_z_cm"] = cluster_pd_2D.planar * 10
        self.cluster_pd_2D = cluster_pd_2D

    def build_tracks_pd(self, subrun_tgt):
        run_l = []
        subrun_l = []
        count_l = []
        x_fit = []
        y_fit = []
        planar_di = {
            "res_planar_0_x": [],
            "res_planar_1_x": [],
            "res_planar_2_x": [],
            "res_planar_3_x": [],
            "res_planar_0_y": [],
            "res_planar_1_y": [],
            "res_planar_2_y": [],
            "res_planar_3_y": []

        }
        for run in (self.cluster_pd_2D["run"].unique()):
            cluster_pd_2D_c0 = self.cluster_pd_2D[self.cluster_pd_2D.run == run]
            if subrun_tgt != None:
                cluster_pd_2D_c0 = cluster_pd_2D_c0[cluster_pd_2D_c0.subrun == subrun_tgt]
            for subrun in cluster_pd_2D_c0["subrun"].unique():
                data_pd_cut_1 = cluster_pd_2D_c0[cluster_pd_2D_c0.subrun == subrun]
                for count in data_pd_cut_1["count"].unique():
                    data_pd_cut_2 = data_pd_cut_1[data_pd_cut_1["count"] == count]
                    if len(data_pd_cut_2) > 3:
                        run_l.append(run)
                        subrun_l.append(subrun)
                        count_l.append(count)
                        fit_x = fit_1_d(data_pd_cut_2.cl_pos_z_cm, data_pd_cut_2.cl_pos_x_cm)
                        fit_y = fit_1_d(data_pd_cut_2.cl_pos_z_cm, data_pd_cut_2.cl_pos_y_cm)
                        x_fit.append(fit_x)
                        y_fit.append(fit_y)
                        for planar in range(0, 4):
                            if len(data_pd_cut_2[data_pd_cut_2.planar == planar]) == 0:
                                planar_di[f"res_planar_{planar}_x"].append(np.nan)
                                planar_di[f"res_planar_{planar}_y"].append(np.nan)
                            else:
                                data_pd_cut_3 = data_pd_cut_2[data_pd_cut_2.planar == planar]
                                planar_di[f"res_planar_{planar}_x"].append(calc_res(data_pd_cut_3.cl_pos_x_cm, fit_x, data_pd_cut_3.cl_pos_z_cm))
                                planar_di[f"res_planar_{planar}_y"].append(calc_res(data_pd_cut_3.cl_pos_y_cm, fit_y, data_pd_cut_3.cl_pos_z_cm))
        dict_4_pd = {
            "run"           : run_l,
            "subrun"        : subrun_l,
            "count"         : count_l,
            "x_fit"         : x_fit,
            "y_fit"         : y_fit,
            "res_planar_0_x": planar_di["res_planar_0_x"],
            "res_planar_1_x": planar_di["res_planar_1_x"],
            "res_planar_2_x": planar_di["res_planar_2_x"],
            "res_planar_3_x": planar_di["res_planar_3_x"],
            "res_planar_0_y": planar_di["res_planar_0_y"],
            "res_planar_1_y": planar_di["res_planar_1_y"],
            "res_planar_2_y": planar_di["res_planar_2_y"],
            "res_planar_3_y": planar_di["res_planar_3_y"]
        }
        return (pd.DataFrame(dict_4_pd))

    def save_tracks_pd(self, subrun="ALL"):
        if subrun == "ALL":
            self.tracks_pd.to_pickle("{}/raw_root/{}/tracks_pd_2D.pickle.gzip".format(self.data_folder, self.run_number), compression="gzip")
        else:
            self.tracks_pd.to_pickle("{}/raw_root/{}/tracks_pd_2D_sub_{}.pickle.gzip".format(self.data_folder, self.run_number, subrun), compression="gzip")

    def load_tracks_pd(self):
        self.tracks_pd = pd.read_pickle("{}/raw_root/{}/tracks_pd_2D.pickle.gzip".format(self.data_folder, self.run_number), compression="gzip")

    def append_tracks_pd(self):
        path = "{}/raw_root/{}/tracks_pd_2D.pickle.gzip".format(self.data_folder, self.run_number)
        if os.path.isfile(path):
            tracks_dataframe_old = pd.read_pickle(path, compression="gzip")
            self.tracks_pd = pd.concat((self.tracks_pd, tracks_dataframe_old))
        self.tracks_pd.to_pickle("{}/raw_root/{}/tracks_pd_2D.pickle.gzip".format(self.data_folder, self.run_number), compression="gzip")

    def read_subruns(self):
        """
        Returns the list of subruns in the run
        :return:
        """
        return (self.cluster_pd_2D.subrun.unique())


class tracking_1d:
    """
    Simple tracking 1D 4 data selection
    """

    def __init__(self, run_number, data_folder, alignment, cylinder):
        """
        It needs the cluster 2_D pickle file to run
        :param alignment:
        :param run_number:
        :param data_folder:
        """
        self.run_number = run_number
        self.data_folder = data_folder
        self.residual_tol = 0.15
        self.alignment = alignment
        self.PUT = False  ## Planar under test
        self.cylinder = cylinder

    def load_cluster_1D(self, cylinder=False):
        """
        Load the cluster 2-D file
        :return:
        """
        cluster_pd_1D = pd.read_feather("{}/raw_root/{}/cluster_pd_1D-zstd.feather".format(self.data_folder, self.run_number))
        if cylinder:
            cluster_pd_1D = cluster_pd_1D[cluster_pd_1D.cl_pos_x.notna()]
        else:
            if not self.alignment:
                cluster_pd_1D["cl_pos_x_cm"] = cluster_pd_1D.cl_pos_x * 0.0650
                cluster_pd_1D["cl_pos_y_cm"] = cluster_pd_1D.cl_pos_y * 0.0650
                cluster_pd_1D["cl_pos_z_cm"] = cluster_pd_1D.planar * 10
            else:
                corr_matrix = self.search_corr_matrix()
                cluster_pd_1D["cl_pos_x_cm"] = cluster_pd_1D.cl_pos_x * 0.0650
                cluster_pd_1D["cl_pos_y_cm"] = cluster_pd_1D.cl_pos_y * 0.0650
                cluster_pd_1D["cl_pos_z_cm"] = cluster_pd_1D.planar * 10
                for planar in (0, 1, 2, 3):
                    for view in ("x", "y"):
                        cluster_pd_1D.loc[cluster_pd_1D.planar == planar, f"cl_pos_{view}_cm"] = cluster_pd_1D.loc[cluster_pd_1D.planar == planar, f"cl_pos_{view}_cm"] - corr_matrix[planar][view]
        cluster_pd_1D = cluster_pd_1D.astype(  ## Verifica che i campi che devono essere interi lo siano
            {"run"    : int,
             "subrun" : int,
             "count"  : int,
             "planar" : int,
             "cl_size": int,
             "cl_id"  : int}
        )
        self.cluster_pd_1D = cluster_pd_1D

        if self.alignment:
            self.save_aligned_clusters()

    def save_aligned_clusters(self, subrun="All"):
        """
        Per salvare il i clusters corretti
        :param subrun:
        :return:
        """
        # TODO sistemare per poterlo runnurare su singolo run
        self.cluster_pd_1D.to_pickle("{}/raw_root/{}/cluster_pd_1D_align.pickle.gzip".format(self.data_folder, self.run_number), compression="gzip")

    def search_corr_matrix(self):
        if os.path.isfile(self.data_folder + "/alignment/" + f"{self.run_number}"):
            return pickle.load(open(self.data_folder + "/alignment/" + f"{self.run_number}", 'rb'))
        else:
            return None

    def fit_tracks_view(self, df, view):
        """
        Builds tracks on 1 view
        :param df:
        :return:
        """
        pd_fit_l = []  ## list of rows to fit
        ids = []
        for planar in df.planar.unique():
            if self.PUT is False or (self.PUT != planar):
                df_p = df[df.planar == planar]  ## select planar
                to_fit = df_p[df_p['cl_charge'] == df_p['cl_charge'].max()]  ## Finds maximum charge cluster

                if len(to_fit) > 1:  ## If we have 2 cluster with the exact same charge...
                    pd_fit_l.append(to_fit.iloc[[0]])
                    ids.append((planar, to_fit.iloc[[0]].cl_id.values[0]))


                else:
                    pd_fit_l.append(to_fit)
                    ids.append((planar, to_fit.cl_id.values[0]))
            else:
                pass
        pd_fit = pd.concat(pd_fit_l)
        fit = fit_1_d(pd_fit.cl_pos_z_cm, pd_fit[f"cl_pos_{view}_cm"])

        res_dict = {}

        pd_fit_l = []  ## list of rows to fit lo rigenero per usarlo per il calcolo del residuo
        ids = []
        for planar in df.planar.unique():
            df_p = df[df.planar == planar]  ## select planar
            to_fit = df_p[df_p['cl_charge'] == df_p['cl_charge'].max()]  ## Finds maximum charge cluster

            if len(to_fit) > 1:  ## If we have 2 cluster with the exact same charge...
                pd_fit_l.append(to_fit.iloc[[0]])
                ids.append((planar, to_fit.iloc[[0]].cl_id.values[0]))

            else:
                pd_fit_l.append(to_fit)
                ids.append((planar, to_fit.cl_id.values[0]))
        pd_fit = pd.concat(pd_fit_l)

        for planar in df.planar.unique():
            pd_fit_pl = pd_fit[pd_fit.planar == planar]
            calc_res(pd_fit_pl[f"cl_pos_{view}_cm"], fit, pd_fit_pl.cl_pos_z_cm)
            res_dict[planar] = calc_res(pd_fit_pl[f"cl_pos_{view}_cm"], fit, pd_fit_pl.cl_pos_z_cm)

        return fit, ids, res_dict

    def build_tracks_pd(self, sub_pd):
        """

        :param subrun_tgt:
        :return:
        """
        run_l = []
        subrun_l = []
        count_l = []
        x_fit = []
        y_fit = []
        planar_di = {
            "res_planar_0_x": [],
            "res_planar_1_x": [],
            "res_planar_2_x": [],
            "res_planar_3_x": [],
            "res_planar_0_y": [],
            "res_planar_1_y": [],
            "res_planar_2_y": [],
            "res_planar_3_y": []
        }
        cl_id_l = []
        n_points = []
        if self.cylinder:
            sub_pd = sub_pd.apply(calc_pos_x_cylinder, 1)

        for count in sub_pd["count"].unique():
            df_c2 = sub_pd[sub_pd["count"] == count]  # df_c2 is shorter

            # Build track X
            df_c2_x = df_c2[df_c2.cl_pos_x_cm.notna()]
            if len(df_c2_x.planar.unique()) > 2:  ## I want at least 3 point in that view
                if self.PUT is False or (len(df_c2_x[df_c2_x.planar != self.PUT].planar.unique()) > 2):
                    fit_x, cl_ids, res_dict = self.fit_tracks_view(df_c2_x, "x")
                    run_l.append(self.run_number)
                    subrun_l.append(int(sub_pd.subrun.mean()))
                    count_l.append(count)
                    x_fit.append(fit_x)
                    y_fit.append(np.nan)
                    for planar in range(0, 4):
                        if planar in res_dict.keys():
                            planar_di[f"res_planar_{planar}_x"].append(res_dict[planar])
                            planar_di[f"res_planar_{planar}_y"].append(np.nan)
                        else:
                            planar_di[f"res_planar_{planar}_x"].append(np.nan)
                            planar_di[f"res_planar_{planar}_y"].append(np.nan)
                    cl_id_l.append(cl_ids)
                    n_points.append(len(df_c2_x.planar.unique()))

            # Build track Y
            df_c2_y = df_c2[df_c2.cl_pos_y_cm.notna()]
            if len(df_c2_y.planar.unique()) > 2:  ## I want at least  3 point in that view
                if self.PUT is False or (len(df_c2_y[df_c2_y.planar != self.PUT].planar.unique()) > 2):
                    fit_y, cl_ids, res_dict = self.fit_tracks_view(df_c2_y, "y")
                    run_l.append(self.run_number)
                    subrun_l.append(int(sub_pd.subrun.mean()))
                    count_l.append(count)
                    x_fit.append(np.nan)
                    y_fit.append(fit_y)
                    for planar in range(0, 4):
                        if planar in res_dict.keys():
                            planar_di[f"res_planar_{planar}_x"].append(np.nan)
                            planar_di[f"res_planar_{planar}_y"].append(res_dict[planar])
                        else:
                            planar_di[f"res_planar_{planar}_x"].append(np.nan)
                            planar_di[f"res_planar_{planar}_y"].append(np.nan)
                    cl_id_l.append(cl_ids)
                    n_points.append(len(df_c2_x.planar.unique()))

        dict_4_pd = {
            "run"           : run_l,
            "subrun"        : subrun_l,
            "count"         : count_l,
            "x_fit"         : x_fit,
            "y_fit"         : y_fit,
            "res_planar_0_x": planar_di["res_planar_0_x"],
            "res_planar_1_x": planar_di["res_planar_1_x"],
            "res_planar_2_x": planar_di["res_planar_2_x"],
            "res_planar_3_x": planar_di["res_planar_3_x"],
            "res_planar_0_y": planar_di["res_planar_0_y"],
            "res_planar_1_y": planar_di["res_planar_1_y"],
            "res_planar_2_y": planar_di["res_planar_2_y"],
            "res_planar_3_y": planar_di["res_planar_3_y"],
            "cl_ids"        : cl_id_l
        }
        return (pd.DataFrame(dict_4_pd))

    def save_tracks_pd(self, subrun="ALL"):
        if not self.alignment:
            name = "tracks_pd_1D"
        else:
            name = "tracks_pd_1D_align"
        if subrun == "ALL":
            self.tracks_pd.to_pickle("{}/raw_root/{}/{}.pickle.gzip".format(self.data_folder, self.run_number, name), compression="gzip")
        else:
            self.tracks_pd.to_pickle("{}/raw_root/{}/{}_sub_{}.pickle.gzip".format(self.data_folder, self.run_number, name, subrun), compression="gzip")

    def load_tracks_pd(self, subrun="ALL"):
        if not self.alignment:
            name = "tracks_pd_1D"
        else:
            name = "tracks_pd_1D_align"

        if subrun == "ALL":
            self.tracks_pd = pd.read_pickle("{}/raw_root/{}/{}.pickle.gzip".format(self.data_folder, self.run_number, name), compression="gzip")
        else:
            self.tracks_pd = pd.read_pickle("{}/raw_root/{}/{}_sub_{}.pickle.gzip".format(self.data_folder, self.run_number, name, subrun), compression="gzip")

    def append_tracks_pd(self):
        if not self.alignment:
            name = "tracks_pd_1D"
        else:
            name = "tracks_pd_1D_align"

        path = "{}/raw_root/{}/{}.pickle.gzip".format(self.data_folder, self.run_number, name)
        if os.path.isfile(path):
            tracks_dataframe_old = pd.read_pickle(path, compression="gzip")
            self.tracks_pd = pd.concat((self.tracks_pd, tracks_dataframe_old))
        self.tracks_pd.to_pickle("{}/raw_root/{}/{}.pickle.gzip".format(self.data_folder, self.run_number, name), compression="gzip")

    def read_subruns(self, from_track=False):
        """
        Returns the list of subruns in the run
        :return:
        """
        if from_track:
            self.load_tracks_pd()
            return (self.tracks_pd.subrun.unique())
        return (self.cluster_pd_1D.subrun.unique())

    def build_select_cl_pd(self, pds):
        """
        Use the track information to select the 1-D clusters
        :return:
        """
        subrun_tgt = pds[0].subrun.mean()
        df_x = self.build_select_cl_pd_view(pds[1], pds[0], "x", subrun_tgt)
        df_y = self.build_select_cl_pd_view(pds[1], pds[0], "y", subrun_tgt)
        cluster_pd_1D_selected = pd.concat([df_x, df_y])
        return cluster_pd_1D_selected

    def build_select_cl_pd_view(self, track_pd, cluster_pd, view, subrun_tgt):
        sel_cd = []
        for planar in range(0, 4):
            y, x, = np.histogram(track_pd[f"res_planar_{planar}_{view}"], bins=400, range=[-0.2, 0.2])
            x_max = (x[y.argmax()])
            track_pd[f"res_planar_{planar}_{view}"] = track_pd[f"res_planar_{planar}_{view}"] - x_max

        track_pd_view = track_pd[pd.notna(track_pd[f"{view}_fit"])]

        for run in track_pd_view.run.unique():
            pd_r = track_pd_view[track_pd_view.run == run]
            if subrun_tgt != None:
                pd_r = pd_r[pd_r.subrun == subrun_tgt]
            for subrun in (pd_r.subrun.unique()):
                pd_s = pd_r[pd_r.subrun == subrun]
                for count in pd_s["count"].unique():
                    pd_c = pd_s[pd_s["count"] == count]
                    tr_pd = pd_c
                    cl_pd = cluster_pd[(cluster_pd["run"] == run) & (cluster_pd["subrun"] == subrun) & (cluster_pd["count"] == count)]
                    for cl_id in tr_pd.cl_ids.values[0]:
                        res_list = [abs(tr_pd[f"res_planar_{i}_{view}"].values[0]) for i in range(0, 4)]
                        if all(np.array(res_list) < self.residual_tol * 1.5):
                            if abs(tr_pd[f"res_planar_{cl_id[0]}_{view}"].values[0]) < self.residual_tol:
                                sel_cd.append(cl_pd[(cl_pd.cl_id == cl_id[1]) & (cl_pd.planar == cl_id[0]) & (cl_pd[f"cl_pos_{view}"] > 0)])
        if len(sel_cd) > 0:
            return pd.concat(sel_cd)
        else:
            return pd.DataFrame()

    def save_sel_cl_pd(self, subrun="ALL"):
        if subrun == "ALL":
            self.cluster_pd_1D_selected.to_pickle("{}/raw_root/{}/sel_cluster_pd_1D.pickle.gzip".format(self.data_folder, self.run_number), compression="gzip")
        else:
            self.cluster_pd_1D_selected.to_pickle("{}/raw_root/{}/sel_cluster_pd_1D_sub_{}_1D.pickle.gzip".format(self.data_folder, self.run_number, subrun), compression="gzip")

    def load_sel_cl_pd(self):
        self.cluster_pd_1D_selected = pd.read_pickle("{}/raw_root/{}/sel_cluster_pd_1D.pickle.gzip".format(self.data_folder, self.run_number), compression="gzip")

    def append_sel_cl_pd(self):
        path = "{}/raw_root/{}/sel_cluster_pd_1D.pickle.gzip".format(self.data_folder, self.run_number)
        if os.path.isfile(path):
            cluster_pd_1D_selected_old = pd.read_pickle(path, compression="gzip")
            self.cluster_pd_1D_selected = pd.concat((self.cluster_pd_1D_selected, cluster_pd_1D_selected_old))
        self.cluster_pd_1D_selected.to_pickle("{}/raw_root/{}/sel_cluster_pd_1D.pickle.gzip".format(self.data_folder, self.run_number), compression="gzip")


class eff_calculator():
    """
    Class to calculate the efficiency, need the aligned clusters and tracks
    """

    def __init__(self, run_number, data_folder, res_trk_max, res_put_max, cut):
        self.res_trk_max = res_trk_max
        self.res_put_max = res_put_max
        self.run_number = run_number
        self.data_folder = f"{data_folder}/raw_root"
        self.out_folder = f"{self.data_folder}/out_eff/{run_number}"
        self.out_folder_meta = f"{self.data_folder}/out_eff/{run_number}"
        if not os.path.isdir(self.out_folder):
            os.mkdir(self.out_folder)
        if not os.path.isdir(self.out_folder_meta):
            os.mkdir(self.out_folder_meta)
        with open(self.out_folder + "/recap.txt", "w") as recap_file:
            recap_file.write("Run {}\n".format(run_number))

    def gaus_fit(self, track_pd_align, planar, view, PUT):
        c1 = R.TCanvas()
        h1 = R.TH1F("h1", "h1", 400, -0.4, 0.4)
        res_list = []
        cut = 1
        for va in track_pd_align[f"res_planar_{planar}_{view}"].values:
            val = va
            if abs(val) < cut:
                h1.Fill(val)
        h1.Fit("gaus", "S")
        res_mean = h1.GetListOfFunctions().FindObject("gaus").GetParameter(1)
        res_std = h1.GetListOfFunctions().FindObject("gaus").GetParameter(2)
        h1.Draw()
        c1.SaveAs(f"{self.out_folder_meta}/{planar}_{view}_PUT_{PUT}.png")
        #     print ("saving hist")
        #     print (f"Fit mean: {res_mean}, std :{res_std}")
        #     print (f"Distr mean: {h1.GetMean()}, std :{h1.GetStdDev()}")

        return res_mean, res_std

    def build_track_pd(self, cluster_pd, PUT):
        tracking_return_list = []

        for run in cluster_pd.run.unique():
            input_list = cluster_pd.subrun.unique()
            tracker = tracking_1d(0, "", True, cylinder=False)
            tracker.PUT = PUT
            tracker.cluster_pd_1D = cluster_pd[cluster_pd.run == run]
            with Pool(processes=cpu_count()) as pool:
                with tqdm(total=len(input_list), desc="tracking", position=0, leave=False) as pbar:
                    for i, x in enumerate(pool.imap_unordered(tracker.build_tracks_pd, input_list)):
                        tracking_return_list.append(x)
                        pbar.update()

        return pd.concat(tracking_return_list)

    def build_select_cl_pd_view(self, cluster_pd, view, residual_max_number, PUT):
        track_pd = self.build_track_pd(cluster_pd, PUT)
        mean, std = self.gaus_fit(track_pd, PUT, view, PUT)
        residual_max = residual_max_number * std
        sel_cd = []
        track_pd_view = track_pd[pd.notna(track_pd[f"{view}_fit"])]
        for run in tqdm(track_pd_view.run.unique(), desc="Run selection", position=0, leave=False):
            pd_r = track_pd_view[track_pd_view.run == run]
            for subrun in tqdm(pd_r.subrun.unique(), desc="Subrun selction", position=0, leave=False):
                pd_s = pd_r[pd_r.subrun == subrun]
                for count in pd_s["count"].unique():
                    pd_c = pd_s[pd_s["count"] == count]
                    tr_pd = pd_c
                    cl_pd = cluster_pd[(cluster_pd["run"] == run) & (cluster_pd["subrun"] == subrun) & (cluster_pd["count"] == count)]
                    for cl_id in tr_pd.cl_ids.values[0]:
                        residual = tr_pd[f"res_planar_{cl_id[0]}_{view}"].values[0]
                        if (residual > (mean - residual_max)) or (residual < (mean + residual_max)):
                            sel_cd.append(cl_pd[(cl_pd.cl_id == cl_id[1]) & (cl_pd.planar == cl_id[0]) & (cl_pd[f"cl_pos_{view}"] > 0)])
        if len(sel_cd) > 0:
            return pd.concat(sel_cd).reindex(), track_pd_view.reindex()
        else:
            return pd.DataFrame(), track_pd_view.reindex()


def fit_1_d(data_series_x, data_series_y):
    """
    Function used by tracking
    :param data_series_x:
    :param data_series_y:
    :return:
    """
    fit_r = np.polyfit(data_series_x, data_series_y, 1)
    return (fit_r)


def calc_res(pos, fit, cl_pos_z_cm):
    """
    Residual calcolation, used by tracking
    :param pos:
    :param fit:
    :param cl_pos_z_cm:
    :return:
    """
    res = pos.values - (fit[1] + fit[0] * cl_pos_z_cm.values)

    return (res[0])


def change_planar(row):
    """
    Change the planar number for the cylinder geometry
    :param row:
    :return:
    """
    if row.planar == 2:
        if row.cl_pos_x > 630:
            row.planar = 3
        else:
            row.planar = 0
    else:
        if row.cl_pos_x > 856 / 2:
            row.planar = 2
        else:
            row.planar = 1
    return row


def calc_pos_x_cylinder(row):
    if row.planar == 2 or row.planar == 1:
        radius = 18.84
        tot_strips = 856
    else:
        radius = 24.34
        tot_strips = 1260
    row["cl_pos_x_cm"] = np.cos(row.cl_pos_x / tot_strips * 2 * np.pi) * radius
    row["cl_pos_z_cm"] = np.sin(row.cl_pos_x / tot_strips * 2 * np.pi) * radius
    row["cl_pos_y_cm"] = 0

    return row


def compress_hit_pd(hits):
    """
    Compress the dataframe memory usage optimizing the datatypes.

    :param hits:
    :return:
    """
    hit_compre = pd.DataFrame()
    hit_compre["channel"] = hits.channel.astype(np.byte)
    hit_compre["tac"] = hits.tac.astype(np.byte)
    hit_compre["tcoarse"] = hits.tcoarse.astype(np.intc)
    hit_compre["ecoarse"] = hits.ecoarse.astype(np.short)
    hit_compre["tfine"] = hits.tfine.astype(np.short)
    hit_compre["efine"] = hits.efine.astype(np.short)
    hit_compre["tiger"] = hits.tiger.astype(np.byte)
    hit_compre["l1ts_min_tcoarse"] = hits.l1ts_min_tcoarse.astype(np.intc)
    hit_compre["delta_coarse"] = hits.delta_coarse.astype(np.intc)
    hit_compre["count"] = hits["count"].astype(np.intc)
    hit_compre["timestamp"] = hits.timestamp.astype(np.intc)
    hit_compre["gemroc"] = hits.gemroc.astype(np.byte)
    hit_compre["runNo"] = hits.runNo.astype(np.intc)
    hit_compre["subRunNo"] = hits.subRunNo.astype(np.intc)
    hit_compre["l1_framenum"] = hits.l1_framenum.astype(np.intc)
    hit_compre["hit_id"] = hits.hit_id.astype(np.intc)
    hit_compre["strip_x"] = hits.strip_x.astype(np.short)
    hit_compre["strip_y"] = hits.strip_y.astype(np.short)
    hit_compre["planar"] = hits.planar.astype(np.byte)
    hit_compre["FEB_label"] = hits.FEB_label.astype(np.short)
    hit_compre["charge_SH"] = hits.charge_SH.astype(np.half)
    return hit_compre


def verifiy_compression_validity(hits, compress_hits):
    """
    Verify that the min and max values of compressed and uncompressd dataframe are compatbile
    :param hits:
    :param compress_hits:
    :return:
    """
    for column in compress_hits.columns:
        assert (abs(hits[column].max() * 0.99) <= abs(compress_hits[column].max()) <= abs(hits[column].max() * 1.01)) or (compress_hits[column].max() == hits[column].max()), f" Max error in column {column}, {hits[column].max()} != {compress_hits[column].max()}"
        assert (abs(hits[column].min() * 0.99) <= abs(compress_hits[column].min()) <= abs(hits[column].min() * 1.01)) or (compress_hits[column].min()) == (hits[column].min()), f" Min error in column {column}, {hits[column].min()} != {compress_hits[column].min()}"
    return 0


def assign_data(data, centers):
    """
    Assign the hit to the nearest center
    """
    dx = np.zeros([len(centers), len(data)])
    for n, center in enumerate(centers):
        dx[n] = (np.abs(data - center))
    labels = dx.argmin(0)
    max_dist = data[dx.min(0).argmax()]
    return labels, max_dist


def manual_kmean(hit_pos, centers):
    """
        Performs a simple 1D kmean algoritm

    """
    for i in range(0, 2000):
        prev_centers = centers.copy()
        labels, max_dist = assign_data(hit_pos, centers)
        for n, center in enumerate(centers):
            hit_pos_this_c = hit_pos[labels == n]
            if len(hit_pos_this_c) > 0:
                centers[n] = hit_pos_this_c.mean()
            else:
                centers[n] = max_dist

        if np.all(prev_centers == centers):
            return (centers, labels)
    print("WARNING kmeans not converged")
    raise Exception("Note convergence error")
def charge_centroid( hit_pos, hit_charge):
    """
    Charge centroid calcolation
    :param hit_pos:
    :param hit_charge:
    :return:
    """
    # hit_charge=np.abs(hit_charge)
    hit_charge[hit_charge < 0.1] = 0.1
    ret_centers = (np.sum([x * c for (x, c) in zip(hit_pos, hit_charge)])) / np.sum(hit_charge)
    return ret_centers

def convert_none(input_):
    try:
        return float(input_)
    except ValueError:
        return float(0)
