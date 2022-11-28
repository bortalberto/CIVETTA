import ROOT as R
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import zscore
import time
def doublegaus(x, a_0, x0_0, sigma_0, a_1, x0_1, sigma_1, c):
    return gaus(x, a_0, x0_0, sigma_0) + gaus(x, a_1, x0_1, sigma_1) + c


def gaus(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))



def root_fit_double_gaus(data, p0, lower_bounds, upper_bounds, sigma_def, nbins=200, mean_0=0):
    """
    Function to fit or double fit gaussian data
    """
    nbins=nbins
    data={"res":data.values.astype(np.float32) }
    rdf = R.RDF.MakeNumpyDataFrame(data)
    amodel=R.RDF.TH1DModel("h1","h1",nbins,mean_0-sigma_def,mean_0+sigma_def)
    h1 = rdf.Histo1D(amodel,"res")
    func=R.TF1("func", "gaus(0) + gaus(3) +[6]", mean_0-sigma_def,mean_0+sigma_def,6)
    a_0, mean_0, sigma_0, a_1, mean_1, sigma_1, c = p0
    func.SetParameters(a_0,mean_0,sigma_0,a_1,mean_1,sigma_1, c)
    for n, limits in enumerate(zip(lower_bounds,upper_bounds)):
        func.SetParLimits (n, limits[0], limits[1])
    gaussFit=h1.Fit(func,"BQ")
    pars=func.GetParameters()
    popt=[pars[i] for i in range (0,7)]
    chi2 = func.GetChisquare()
    ndof = func.GetNDF ()
    return popt, chi2


def double_gaus_fit_root(tracks_pd, view="x", put=-1, sigma_def=0.2, pl_list=range(0,4)):
    """
    Performs a gaussian double fit
    :param tracks_pd:
    :param view:
    :param put: planar to be excluded from the fit
    :param sigma_def:
    :return:
    """
    popt_list = []
    pcov_list = []
    res_list = []
    R_list = []
    chi_list = []
    deg_list = []
    for pl in pl_list:
        if pl==put:
            popt_list.append(0)
            pcov_list.append(0)
            res_list.append(0)
            R_list.append(1)
            chi_list.append(1)
            deg_list.append(1)
        else:
            data = tracks_pd[f"res_{view}"].apply(lambda x: x[pl])
            data = data[abs(data) < 10]
            # print (data.describe())
            sigma_def = estimate_sigma_def(data)
            # print ("---")
            # print (view, pl)
            # print (np.median(data))
            # print (sigma_def)
            # print ("---")
            # print (sigma_def)
            data = data[abs(data - np.median(data)) < sigma_def]
            nbins=200
            if len(data)< 2000:
                nbins=100
            y, x = np.histogram(data, bins=nbins, range=[np.median(data)-sigma_def,np.median(data)+sigma_def])

            x = (x[1:] + x[:-1]) / 2
            # x = np.insert(x,0,-0.2)
            # y = np.insert(y,0,0)
            #             x=x[4000:6000]
            #             y=y[4000:6000]
            mean_1 =  x[np.argmax(y)]
            mean_0 =  x[np.argmax(y)]
            a_0 = np.max(y) -1
            a_1 = np.max(y) / 10
            sigma_0 = sigma_def/10
            sigma_1 = sigma_def
            c=1
#             lower_bound=[0, x[np.argmax(y)]-0.01,0,0,x[np.argmax(y)]-0.01,0,0]
#             upper_bound=[np.inf,  x[np.argmax(y)]+0.01, 1, np.inf,x[np.argmax(y)]+0.01,2,100]
#             popt, pcov = curve_fit(doublegaus, x, y,sigma=error,p0=[a_0, mean_0, sigma_0, a_1, mean_1, sigma_1, c], bounds=(lower_bound, upper_bound))

            lower_bound=[np.max(y)/5*4,mean_0-sigma_0/3,       0,             0,mean_0-sigma_0/3,            sigma_def/10,     0]
            upper_bound=[np.max(y)    ,mean_0+sigma_0/3, sigma_0,   np.max(y)/5,mean_0+sigma_0/3,sigma_def + sigma_def/10,     200]
            guess=[a_0, mean_0, sigma_0 - sigma_0/10, a_1, mean_1, sigma_1, c]
            # print(pl)
            # print(lower_bound)
            # print ([a_0, mean_0, sigma_0, a_1, mean_1, sigma_1, c])
            # print (upper_bound)
            # print ("---")
            # print ("Fit param")
            # print (guess, lower_bound, upper_bound, sigma_def )
            popt, chi_sqr = root_fit_double_gaus(data, guess, lower_bound, upper_bound, sigma_def, nbins=nbins)
            # print ("fitted")
            pcov=0
            popt_list.append(popt)
            pcov_list.append(pcov)
            yexp = doublegaus(x, *popt)
            ss_res = np.sum((y - yexp) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            res_list.append(y - yexp)
            r2 = 1 - (ss_res / ss_tot)  # ynorm= 1000*y/np.sum(y)
            R_list.append(r2)
#             print(scipy.stats.chisquare(y, yexp,len(x)-6-1))

#             chi_list.append(scipy.stats.chisquare(y, yexp,len(x)-6-1))
#             chi_list.append(np.divide(np.square(y - yexp), yexp) * (np.sqrt(y))/np.sqrt(len(data))) #with weigth
            chi_list.append(chi_sqr)


            deg_list.append(len(x)-6-1)
    #         yexp=doublegaus(x, *popt)
    #         y_exp_norm =1000*yexp/np.sum(yexp)
    #         print (np.sum(ynorm))
    #         print (np.sum(y_exp_norm))
    #         print (chisquare(ynorm,y_exp_norm, 6 ))
    return popt_list, pcov_list, res_list, R_list, chi_list, deg_list


def single_root_fit(data, p0, lower_bounds, upper_bounds, sigma_def=0.2):
    nbins=200
    mean = np.mean(data.values.astype(np.float32))
    data = {"res": data.values.astype(np.float32)}
    if len(data) < 2000:
        nbins = 100
    rdf = R.RDF.MakeNumpyDataFrame(data)
    amodel = R.RDF.TH1DModel("h1", "h1", nbins, mean - sigma_def, mean + sigma_def)
    h1 = rdf.Histo1D(amodel, "res")
    func = R.TF1("func", "gaus(0) +[3]", mean - sigma_def, mean + sigma_def, 4)
    a_0, mean_0, sigma_0, c = p0
    func.SetParameters(a_0, mean_0, sigma_0, c)
    for n, limits in enumerate(zip(lower_bounds, upper_bounds)):
        func.SetParLimits(n, limits[0], limits[1])
    gaussFit = h1.Fit(func, "BQ")
    pars = func.GetParameters()
    errors = func.GetParErrors()
    error = [ errors[i] for i in range (0,4)]
    popt = [pars[i] for i in range(0, 4)]
    chi2 = func.GetChisquare()
    ndof = func.GetNDF()
    return popt, chi2, error, ndof

def plot_residuals_single_gauss(cl_pd_res, view, popt_list, R_list, pl, chi_list, deg_list, sigma_def=0.2):
        data = cl_pd_res
        # sigma_def = estimate_sigma_def(data)

        data = data[abs(data - np.median(data)) < sigma_def]
        nbins = 200
        if len(data) < 2000:
            nbins = 100
        y, x = np.histogram(data, bins=nbins, range=[np.median(data) - sigma_def, np.median(data) + sigma_def])
        x = (x[1:] + x[:-1]) / 2
        # x = np.insert(x, 0, -0.2)
        # y = np.insert(y, 0, 0)
        popt = popt_list
        f, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, 'b*', label='data')
        x = np.arange(np.min(x), np.max(x), 0.0002)
        ax.plot(x, gaus(x, *popt[0:3]) + popt[3] , 'c-', label='fit 0')
        # ax.plot(x, gaus(x, *popt[3:6]), 'g-', label='fit 1')
        # ax.plot(x, doublegaus(x, *popt), 'r-', label='fit cumulative')
        ax.grid()
        # plt.legend()
        # plt.title('Fig. 3 - Fit for Time Cons§tant')
        ax.set_ylabel('#')
        ax.set_xlabel('Residual [cm]')
        # plt.ion()
        # plt.show()
        ax.set_title(f"Fit view {view}, planar{pl}")
        ax.text(y=np.max(y) * 0.7, x=0 + popt[2],
                s=f"R^2={R_list:.4f}\nNorm_0={popt[0]:.2f}, Mean_0={popt[1] * 10000:.2f}um, Sigma_0={(popt[2]) * 10000:.2f}um"
                  f"\nChi_sqrt={chi_list:.3e}, Chi_sqrt/NDoF = {chi_list / deg_list:.3e}",
                fontsize="small")
        ax.set_xlim([np.min(x), np.max(x)])
        #     if put==pl:
        #         plt.savefig(os.path.join(os.path.join(path_out_eff, "res_fit"), f"fit_res_DUT_pl{pl}_DUT_{put}{view}.png"))
        #     else:
        #         plt.savefig(os.path.join(os.path.join(path_out_eff, "res_fit"), f"fit_res_TRK_pl{pl}_DUT_{put}{view}.png"))

        return f, ax


def single_gaus_fit_root(cl_pd_res, sigma_def=0.4):
    data = cl_pd_res
    # data = data[abs(data - np.median(data)) < sigma_def]
    nbins = 200
    if len(data) < 2000:
        nbins = 100
    y, x = np.histogram(data, bins=nbins, range=[np.median(data) - sigma_def, np.median(data) + sigma_def])
    x = (x[1:] + x[:-1]) / 2
    # x = np.insert(x, 0, -0.2)
    # y = np.insert(y, 0, 0)
    #             x=x[4000:6000]
    #             y=y[4000:6000]
    mean_0 = x[np.argmax(y)]
    a_0 = np.max(y)
    sigma_0 = np.std(data)
    c = 1
    #             lower_bound=[0, x[np.argmax(y)]-0.01,0,0,x[np.argmax(y)]-0.01,0,0]
    #             upper_bound=[np.inf,  x[np.argmax(y)]+0.01, 1, np.inf,x[np.argmax(y)]+0.01,2,100]
    #             popt, pcov = curve_fit(doublegaus, x, y,sigma=error,p0=[a_0, mean_0, sigma_0, a_1, mean_1, sigma_1, c], bounds=(lower_bound, upper_bound))
    guess=[a_0, mean_0, sigma_0, c]
    lower_bound = [0, x[np.argmax(y)] - 0.01, 0, 0]
    upper_bound = [np.max(y)+10, x[np.argmax(y)] + 0.01, sigma_0*2, 200]
    # print ("Fit param")
    # print (guess, lower_bound, upper_bound, sigma_def )

    popt, chi_sqr, error, ndof = single_root_fit(data, guess,
                                    lower_bound, upper_bound, sigma_def=sigma_def)
    # print ("fitted")

    pcov = 0
    yexp = gaus(x, *popt[0:3]) + popt[3]
    ss_res = np.sum((y - yexp) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    res=(y - yexp)
    r2 = 1 - (ss_res / ss_tot)  # ynorm= 1000*y/np.sum(y)
    #             print(scipy.stats.chisquare(y, yexp,len(x)-6-1))

    #             chi_list.append(scipy.stats.chisquare(y, yexp,len(x)-6-1))
    #             chi_list.append(np.divide(np.square(y - yexp), yexp) * (np.sqrt(y))/np.sqrt(len(data))) #with weigth

    deg = ndof
    #         yexp=doublegaus(x, *popt)
    #         y_exp_norm =1000*yexp/np.sum(yexp)
    #         print (np.sum(ynorm))
    #         print (np.sum(y_exp_norm))
    #         print (chisquare(ynorm,y_exp_norm, 6 ))
    return popt, pcov, res, r2, chi_sqr, deg, error


def plot_residuals(tracks_pd_res, view,popt_list,R_list, path_out_eff, put,put_mean, put_sigma,nsigma_eff, pl, chi_list, deg_list, chi_sq_trackers = False):
    data = tracks_pd_res[f"res_{view}"].apply(lambda x: x[pl])
    data = data[abs(data) < 10]
    sigma_0 = estimate_sigma_def(data)
    # data = data[abs(data - np.median(data)) < sigma_def*2]
    nbins = 200
    if len(data) < 2000:
        nbins = 100
    # print ("a")
    y, x = np.histogram(data, bins=nbins, range=[np.median(data)-sigma_0, np.median(data)+sigma_0])
    x = (x[1:] + x[:-1]) / 2
    # x = np.insert(x, 0, -0.2)
    # y = np.insert(y, 0, 0)
    if len (popt_list)>1:
        popt = popt_list[pl]
    else:
        popt = popt_list[0]
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b*', label='data')
    x = np.arange(np.min(x),np.max(x), 0.0002)
    plt.plot(x, gaus(x, *popt[0:3]), 'c-', label='fit 0')
    plt.plot(x, gaus(x, *popt[3:6]), 'g-', label='fit 1')
    plt.plot(x, doublegaus(x, *popt), 'r-', label='fit cumulative')
    plt.grid()
    # plt.legend()
    # plt.title('Fig. 3 - Fit for Time Cons§tant')
    plt.ylabel('#')
    plt.xlabel('Residual [cm]')
    # plt.ion()
    # plt.show()
    plt.title(f"Fit view {view}, DUT= {put}, planar{pl}")
    plt.text(y=np.max(y)*0.7, x=np.min(x)+0.001, s=f"R^2={R_list[pl]:.4f}\nNorm_0={popt[0]:.2f}, Mean_0={popt[1]*10000:.2f}um, Sigma_0={(popt[2])*10000:.2f}um"
                                                           f"\n Norm_1={popt[3]:.2f}, Mean_1={popt[4]*10000:.2f}um, Sigma_1={abs(popt[5])*10000:.2f}um"
                                                           f"\n Chi_sqrt={chi_list[pl]:.3e}, Chi_sqrt/NDoF = {chi_list[pl]/deg_list[pl]:.3e}", fontsize="small")
    if not chi_sq_trackers:
        plt.plot([put_mean + nsigma_eff * put_sigma, put_mean + nsigma_eff * put_sigma], [0, np.max(y)], 'r-.')
        plt.plot([put_mean - nsigma_eff * put_sigma, put_mean - nsigma_eff * put_sigma], [0, np.max(y)], 'r-.')
    plt.xlim(np.min(x), np.max(x))
    if len(path_out_eff)>1:
        if put==pl:
            plt.savefig(os.path.join(os.path.join(path_out_eff, "res_fit"), f"fit_res_DUT_pl{pl}_DUT_{put}{view}.png"))
        else:
            plt.savefig(os.path.join(os.path.join(path_out_eff, "res_fit"), f"fit_res_TRK_pl{pl}_DUT_{put}{view}.png"))
    else:
        plt.show()
    plt.close()


def estimate_sigma_def(data):
    data = data[abs(data) < 2]
    # print (f"std {std}")
    # print (f"data:  ({len(data)})")
    z_scores = zscore(data)
    # std = np.std(data[np.abs(z_scores) < 2])
    data=data[np.abs(z_scores) < 2]
    std= np.std(data)
    # print (f"std {std}")
    # print (f"data:  ({len(data)})")
    # z_scores = zscore(data)
    # std = np.std(data[np.abs(z_scores) < 1])
    # data=data[np.abs(z_scores) < 1]
    # # print (f"std {std}")
    # # print (f"data:  ({len(data)})")
    # z_scores = zscore(data)
    # std = np.std(data[np.abs(z_scores) < 1])
    # data=data[np.abs(z_scores) < 1]
    # print (f"std {std}")
    # print (f"data:  ({len(data)})")
    # z_scores = zscore(data)
    # std = np.std(data[np.abs(z_scores) < 1])
    # data=data[np.abs(z_scores) < 1]
    # print (f"std {std}")
    # print (f"data:  ({len(data)})")
    popt_list, pcov_list, res_list, R_list, chi, deg_list, error = single_gaus_fit_root(data, std * 5)
    if std > 5*popt_list[2]:
        std = popt_list[2]
        popt_list, pcov_list, res_list, R_list, chi, deg_list, error = single_gaus_fit_root(data, std * 5)

    # f, ax = plot_residuals_single_gauss(data, "x", popt_list, R_list, 2, chi, deg_list, std*5)
    # f.savefig(f"/media/disk2T/VM_work_zone/data/perf_out/403/res_fit/{time.time()}.png")
    # plt.close(f)
    # print (popt_list[2]*7)
    return (abs(popt_list[2]*10))
