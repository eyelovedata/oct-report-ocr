"""
Author: Youchen (Victor) Zhang
Email: youchenz@stanford.edu

Updated by Dec 1st,2022
"""

from google.cloud import bigquery
from google.cloud import storage
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from IPython.display import HTML
import numpy as np
import os
from datetime import *
import pandas as pd
import seaborn as sns
from collections import Counter
from collections import OrderedDict
import pyCompare
from patsy.contrasts import Treatment
from sklearn import preprocessing
import statsmodels.api as sm
from scipy import stats
import warnings
import re
from tqdm import tqdm
import pydicom
import cv2
import PIL 
import csv
from pathlib import Path
import random
from collections import defaultdict
from pdf2image import convert_from_path
from paddleocr import PaddleOCR,draw_ocr
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import shutil
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

warnings.filterwarnings("ignore")
plt.rcParams['animation.embed_limit'] = 2**128
plt.rcParams["figure.figsize"] = (22,16)
client=bigquery.Client(project='som-nero-phi-sywang-starr')


exam_dict = {"EPIC#OPH601":"cctod",
             "EPIC#OPH602":"cctos",
             "EPIC#OPH603":"cctdate",
             "EPIC#OPH676":"gonodtemp",
             "EPIC#OPH677":"gonodnas",
             "EPIC#OPH678":"gonodsup",
             "EPIC#OPH679":"gonodinf",
             "EPIC#OPH680":"gonostemp",
             "EPIC#OPH681":"gonosnas",
             "EPIC#OPH682":"gonossup",
             "EPIC#OPH683":"gonosinf",
             #"EPIC#OPH151":"tmethod",
             #"EPIC#OPH152": "ttime", 
             "EPIC#OPH153": "tod",
             "EPIC#OPH154": "tos",
             "EPIC#OPH1090": "feodcdr",
             "EPIC#OPH1091": "feoscdr",
             "EPIC#OPH1051": "feoddisc",
              "EPIC#OPH1052": "feodmac",
              "EPIC#OPH1053": "feodperiph",
              "EPIC#OPH1054": "feodvess",
              "EPIC#OPH1055": "feoddisc",
              "EPIC#OPH1056": "feosmac",
              "EPIC#OPH1057": "feosperiph",
              "EPIC#OPH1058": "feosvess",
              "EPIC#OPH1011": "sleodll",
             "EPIC#OPH1012": "sleodcs",
             "EPIC#OPH1013":"sleodk", 
             "EPIC#OPH1014": "sleodac",
             "EPIC#OPH1015": "sleodlens",
             "EPIC#OPH1016": "sleodiris",
             "EPIC#OPH1017": "sleodvit",
             "EPIC#OPH1018": "sleosll", 
             "EPIC#OPH1019": "sleoscs",
             "EPIC#OPH1020": "sleosk", 
             "EPIC#OPH1021": "sleosac", 
             "EPIC#OPH1022": "sleoslens", 
              "EPIC#OPH1023": "sleoslens", 
              "EPIC#OPH1024": "sleosvit",
             "EPIC#OPH101": "vamethod", 
             "EPIC#OPH102": "vacorr", 
             "EPIC#OPH103": "vaoddistsc",
             "EPIC#OPH104": "vaoddistcc",
             "EPIC#OPH105": "vaoddistscph", 
             "EPIC#OPH106": "vaodnearsc", 
             "EPIC#OPH107": "vaodnearcc", 
             "EPIC#OPH108": "vaosdistsc", 
             "EPIC#OPH109": "vaosdistcc", 
             "EPIC#OPH110": "vaosdistscph",
             "EPIC#OPH111": "vaosnearsc",
             "EPIC#OPH112": "vaosnearcc", 
             "EPIC#OPH117": "vaoddistccph",
             "EPIC#OPH118": "vaosdistccph", 
             "EPIC#OPH201": "wrxodsph",
             "EPIC#OPH202": "wrxodcyl",
             "EPIC#OPH203": "wrxodax", 
             "EPIC#OPH204": "wrxodadd",
             "EPIC#OPH207": "wrxossph",
             "EPIC#OPH208": "wrxoscyl",
             "EPIC#OPH209": "wrxosax",
             "EPIC#OPH210": "wrxosadd",
             "EPIC#OPH213": "wrxage",
            "EPIC#OPH214": "wrxtype",
            "EPIC#OPH251": "mrxodsph",
             "EPIC#OPH252": "mrxodcyl",
             "EPIC#OPH253": "mrxodax",
             "EPIC#OPH254": "mrxodadd",
             "EPIC#OPH255": "mrxodva",
             "EPIC#OPH256": "mrxossph",
             "EPIC#OPH257": "mrxoscyl",
             "EPIC#OPH258": "mrxosax",
             "EPIC#OPH259": "mrxosadd", 
             "EPIC#OPH260": "mrxosva",
             "EPIC#OPH261": "mrxauto",
             "EPIC#OPH1680": "finalrxodva",
             "EPIC#OPH1682": "finalrxosva",
            }

def count_img_dimension(bucket_name, dicomfilepath, delete_file_after_use=True):
    """
    Count frequency of image dimension
    
    Args:
        bucket_name: A bucket where dcm file located in gcloud storage.
        dicomfilepath: A list of dcm file paths obtained from bigquery table 'patient_details' .
        delete_file_after_use: A switch for users to decide if they want to delete dcm file after using.
    
    Example:
        dicomfilepath = df.dicomfilepath
        count_img_dimension('stanfordoptimagroup',dicomfilepath)
        tmp_dict = defaultdict(list)
        for idx, filepath in enumerate(df.loc[(df.studydescription=='OCT (RNFL) - OU') & (df.seriesdescription=='Volume IR')].dicomfilepath):
            shape, dicmfilepath = count_img_dimension('stanfordoptimagroup',filepath)
            tmp_dict[shape].append(dicmfilepath)
    """
    
    source_blob_name = dicomfilepath
    output_file_name = 'plot_image'+'/' + ('/').join(str(source_blob_name).split('/')[4:])
    if not os.path.exists(output_file_name):
        download_blob(bucket_name,source_blob_name, output_file_name, False)
    ds = pydicom.dcmread(output_file_name)
    shape = None

    try:
        shape = ds.pixel_array.shape
        
    except:
        print('Float Pixel Data or Double Float Pixel Data must be present in the dataset')
        
    if delete_file_after_use:
        if os.path.exists(output_file_name):
              os.remove(output_file_name)
        else:
              print("The file does not exist")
    return shape, dicomfilepath


def plot_image(bucket_name, dicomfilepath, delete_file_after_display=True, steps=10, metadata=False):
    """
    Display Dicom Image
    
    Args:
        bucket_name: A bucket where dcm file located in gcloud storage.
        dicomfilepath: A list of dcm file paths obtained from bigquery table 'patient_details' .
        delete_file_after_display: A switch for users to decide if they want to delete dcm file after plotting.
    
    Example:
        dicomfilepath = df.dicomfilepath
        plot_image('stanfordoptimagroup',dicomfilepath)
    """
    ani = None
    source_blob_name = dicomfilepath
    output_file_name = 'plot_image'+'/' + ('/').join(str(source_blob_name).split('/')[4:])
    if not os.path.exists(output_file_name):
        download_blob(bucket_name,source_blob_name, output_file_name)
    ds = pydicom.dcmread(output_file_name)

    try:
        print(ds.pixel_array.shape)
        if len(ds.pixel_array.shape)==3 and ds.pixel_array.shape[2]!=3:
            ani = animate_NxNxN_pixel_array(ds,steps)
        else:    
            plt.imshow(ds.pixel_array)
    except:
        print('Float Pixel Data or Double Float Pixel Data must be present in the dataset')
    
    if metadata:
        print(ds)

    if delete_file_after_display:
        if os.path.exists(output_file_name):
              os.remove(output_file_name)
        else:
              print("The file does not exist")
    return ani


def random_plot_image(bucket_name, dicomfilepath, delete_file_after_display=True, steps=10):
    """
    Randomly display slit lamp image
    
    Args:
        bucket_name: A bucket where dcm file located in gcloud storage.
        dicomfilepath: A list of dcm file paths obtained from bigquery table 'patient_details' .
        delete_file_after_display: A switch for users to decide if they want to delete dcm file after plotting.
    
    Example:
        dicomfilepath = df.dicomfilepath
        random_plot_image('stanfordoptimagroup',dicomfilepath)
    """
    ani = None
    source_blob_name = random.choice(dicomfilepath)
    output_file_name = 'dcm'+'/' + ('/').join(str(source_blob_name).split('/')[4:])
    if not os.path.exists(output_file_name):
        download_blob(bucket_name,source_blob_name, output_file_name)
    ds = pydicom.dcmread(output_file_name)
    
    try:
        print(ds.pixel_array.shape)
        if len(ds.pixel_array.shape)==3 and ds.pixel_array.shape[2]!=3:
            ani = animate_NxNxN_pixel_array(ds,steps)
        else:    
            plt.imshow(ds.pixel_array)
    except:
        print('Float Pixel Data or Double Float Pixel Data must be present in the dataset')
        
    print(ds)

    if delete_file_after_display:
        if os.path.exists(output_file_name):
              os.remove(output_file_name)
        else:
              print("The file does not exist")
    return ani


def animate_NxNxN_pixel_array(ds, steps=10):
    """
    Animate NxNxN pixel array extracted from dcm file
    
    Args:
        ds: pydicom.dcmread('file path')
    
    Return:
        ani: matplotlib.animation.ArtistAnimation
    
    Example:
        ani = animate_NxNxN_pixel_array(ds)
        HTML(ani.to_jshtml())
    
    Reference:
        https://www.kaggle.com/code/pranavkasela/interactive-and-animated-ct-scan/notebook
    """

    fig = plt.figure()

    ims = []
    for image in range(0,ds.pixel_array.shape[0],steps):
        im = plt.imshow(ds.pixel_array[image,:,:], 
                        animated=True, cmap=plt.cm.bone)
        plt.axis("off")
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,
                                    repeat_delay=1000)

    plt.close()
    return ani

def plot_countplot(df, column_name, title, *args, **kwargs):
    """Plots countplot.
    
    Show the counts of observations in each categorical bin using bars.
    
    Args:
        df: A pandas dataframe.
        column_name: A column name in pandas dataframe.
        title: A text string to set plot's title
        *args: Non Keyword Arguments.
        **kwargs: Keyword Arguments.
    
    Examples:
        plot_countplot(df, 'raceth','Raceth countplot')
    
    Sources:
        https://seaborn.pydata.org/generated/seaborn.countplot.html
    """
    
    ax = sns.countplot(x=column_name, data=df, *args, **kwargs)
    ax.tick_params(axis='x', labelrotation=0)
    ax.set(title=title)
    x = Counter(df)
    ax.bar_label(container=ax.containers[0]);


def addlabels(x, y):
    """Add labels in plot.
    
    Add values on top of each bar.
    
    Args:
        x: keys of a Counter object.
        y: values of a Counter object.
    
    Examples:
        iop_counter = Counter(df['tonopen_iop'] - df['applanation_iop'])
        addlabels(list(iop_counter.keys()), list(iop_counter.values()))
    """
    
    for i in range(len(x)):
        plt.text(x[i], y[i]+20, y[i], ha='center')


def cct_scatter_plot(applanation_iop, tonopen_iop ,cct, *args, **kwargs):
    """Plots central cornea thickness scatter plot.
    
    Show the counts of observations in each categorical bin using bars.
    
    Args:
        applanation_iop: GAT IOP readings column of a given dataframe (pd.series).
        tonopen_iop: TP IOP readings column of a given dataframe (pd.series).
        cct: CCT readings column of a given dataframe (pd.series)
        *args: Non Keyword Arguments.
        **kwargs: Keyword Arguments.
    
    Examples:
        cct_scatter_plot(df['applanation_iop'], df['tonopen_iop'] ,df['cct'], alpha=0.4, color='teal')
    """
    
    applanation_iop     = np.asarray(applanation_iop)
    tonopen_iop         = np.asarray(tonopen_iop)
    cct                 = np.asarray(cct) * 25            # Multiply 25 to make it back to normal range for visualization purpose
    diff                = tonopen_iop - applanation_iop
    md                  = np.mean(diff)                   # Mean of the difference
    sd                  = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(cct, diff, *args, **kwargs)
    plt.title('Central Cornea Thickness Scatter Plot', size=25)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.ylabel('Difference (TP - GAT mmHg)',size=18)
    plt.text(max(cct)*1.08,md - 1.96*sd,f'-1.96 SD {md - 1.96*sd:.2f}',fontsize='large')
    plt.text(max(cct)*1.08,md,f'Mean {md:.2f}',fontsize='large')
    plt.text(max(cct)*1.08,md + 1.96*sd, f'+1.96 SD {md + 1.96*sd:.2f}',fontsize='large')
    plt.xlabel('Central Cornea Thickness Readings (μm)', size=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    print('standard deviation: ',sd)


def plot_linear_reg_line(df, x_column, y_column,xlabel,ylabel, title, axhline=False):
    """Plot data and a linear regression model fit.
    
    Plot the relationship between two variables in a DataFrame.
    
    Args:
        df: A pandas dataframe.
        x_column: X variable.
        y_column: Y variable.
        xlabel: X variable's label.
        ylabel: Y variable's label.
        title: A string text for plot's title.
        axhline: Set a gray dash line at the y value is 0.
    
    Examples:
        plot_linear_reg_line(df,"applanation_iop",'diff', 
        'Goldmann Applanation Tonometry (mmHg)','Difference (TP - GAT mmHg)',
        'GAT vs. Difference (TP - GAT mmHg)',axhline=True)
    """
    
    # get coeffs of linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_column],df[y_column])

    # use line_kws to set line label for legend
    ax = sns.regplot(x=x_column, y=y_column, data=df, color='b',
    line_kws={'label':"y={0:.1f}x+{1:.1f}".format(slope,intercept)})
    
    # plot legend
    ax.legend()
    ax.set_xlabel(xlabel,size=18)
    ax.set_ylabel(ylabel, size=18)
    ax.set_title(title, size=25)
    plt.setp(ax.get_legend().get_texts(), fontsize='20') 
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_xticklabels(np.append(ax.get_xticks(),max(df[x_column])+10),size=14)
    ax.set_yticklabels(np.append(ax.get_yticks(),max(df[y_column])+10),size=14)
    ax.axis(xmin=min(df[x_column]),xmax=max(df[x_column])+5)
    ax.axis(ymin=min(df[y_column]),ymax=max(df[y_column])+5)
    
    if axhline is True:
        ax.axhline(0,color='gray', linestyle='--')
    plt.show()

    if p_value < 0.001:
        print('p value is less than 0.001')
    else:
        print('p value is greater than 0.001')
    print('p value:',p_value)
    print('r value:',r_value)
    print('slope:', slope)
    print('intercept:',intercept)
    print('standard deviation err', std_err)
    

def iop_abs_table(df, first_measurement,second_measurement):
    """Plot a table for IOP absolute difference between 2 measurements.
    
    Summarize the count and percentage for each IOP absolute difference
    between two different measurements.
    
    Args:
        df: A pandas dataframe.
        first_measurement: IOP readings measured by GAT.
        second_measurement: IOP readings measured by a instrument that is different from GAT.
    
    Examples:
        iop_abs_table(df, 'applanation_iop','tonopen_iop')
    """
    
    df['abs_diff'] = abs(df[first_measurement] - df[second_measurement])
    table_data = [[sum(df.abs_diff < 2), sum(df.abs_diff < 2)/len(df) * 100],
                [sum(df.abs_diff == 2), sum(df.abs_diff == 2)/len(df)* 100],
                [sum(df.abs_diff == 3), sum(df.abs_diff == 3)/len(df)* 100],
                [sum(df.abs_diff == 4), sum(df.abs_diff == 4)/len(df)* 100],
                [sum(df.abs_diff == 5), sum(df.abs_diff == 5)/len(df)* 100],
                [sum(df.abs_diff == 6), sum(df.abs_diff == 6)/len(df)* 100],
                [sum(df.abs_diff > 6), sum(df.abs_diff > 6)/len(df)* 100]]
    fig, ax =plt.subplots(1,1)
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=table_data,colLabels=['N','%'], rowLabels=['<2', 2,3,4,5,6,'>6'], loc='center')


def sqlpull(client, project_id, dataset_id, table_id): 
    """Fetch data from bigquery table.
    
    Fetch data from bigquery table stored in gcloud
    and save it as a pandas dataframe.
    
    Args:
        client: Bigquery client.
        project_id: A id that tells which project the table is stored under.
        dataset_id: A id that tells which dataset the table is stored under that project.
        table_id: A id that assigned to that table (a.k.a table's file name).
    
    Examples:
        project_id = 'som-nero-phi-sywang-starr'
        dataset_id = 'youchen_data_folder'
        table_id = 'sf_oph_enc_exam_02_23_iop_cct_glauc_priority_corrected'
        df = sqlpull(client,project_id,dataset_id, table_id=table_id)
    
    Returns:
        A dataframe converted from bigquery table.
    """            
    project_id = project_id
    dataset_id = dataset_id
    query="""SELECT *
    FROM `{project_id}.{dataset_id}.{table_id}`
    """.format_map({'project_id': project_id,
                    'dataset_id': dataset_id,
                    'table_id': table_id})
    query_job =client.query(query)
    df=query_job.to_dataframe()
    df.columns = map(str.lower, df.columns)
    return df 


#sql table for glauc_codes
def sqlpull_glauc_codes(outputfolder,
                        outputprojectname, 
                        project_id = 'som-nero-phi-naras-ric', 
                        dataset_id = 'Ophthalmology_tables', 
                        table_id = 'sf_oph_dx_dm'): 
    
    """Fetch glaucoma codes data from bigquery table.
    
    Fetch data from bigquery table stored in gcloud
    and save it as a pandas dataframe.
    
    Args:
        outputfolder: A local directory you choose to store the csv file generated by this function.
        outputprojectname: A project id you created on bigquery that stores the table generated by this function.
        project_id: A id that tells which project the table is stored under.
        dataset_id: A id that tells which dataset the table is stored under that project.
        table_id: A id that assigned to that table (a.k.a table's file name).
    
    Examples:
        sqlpull_glauc_codes(outputfolder, outputprojectname)
    
    Returns:
        A dataframe converted from bigquery table.
    """   
    
    
    project_id = project_id
    dataset_id = dataset_id
    query="""
    SELECT *, 'glaucsec' as glauc_category, 1 as glauc_priority
    FROM `{project_id}.{dataset_id}.{table_id}` 
    Where 
    # secondary glaucoma icd codes sophia provided
    lower(icd10) like '%h40.04%' or lower(icd10) like '%h40.3%' 
    or lower(icd10) like '%h40.4%' or lower(icd10) like '%h40.5%' 
    or lower(icd10) like '%h40.6%' or lower(icd10) like '%h40.81%' 
    or lower(icd10) like '%h40.82%'or lower(icd10) like '%h40.83%' 
    or lower(icd10) like '%h40.13%' or lower(icd10) like '%h40.14%'
    or lower(icd10) like '%q150%' 

    or lower(icd9) like '%365.03%'or lower(icd9) like '%365.3%' 
    or lower(icd9) like '%365.4%' or lower(icd9) like '%365.5%' 
    or lower(icd9) like '%365.6%' or lower(icd9) like '%365.81%'
    or lower(icd9) like '%365.82%'or lower(icd9) like '%365.83%'
    or lower(icd9) like '%365.13%'or lower(icd9) like '%365.14%'

    # when icd10 is null, I used sophia's list of icd9 to label them
    or lower(icd9) = '365.31'
    or lower(icd9) = '365.43' or lower(icd9) = '365.44' 
    or lower(icd9) = '365.52' or lower(icd9) = '365.62' 
    or lower(icd9) = '365.63' or lower(icd9) = '365.65' 

    # add false labeled dx_id code to secondary glaucoma category
    or dx_id = 1595309 or dx_id = 1682364
    or dx_id = 1686726 or dx_id = 1352937
    or dx_id = 1595472 or dx_id = 1571606
    or dx_id = 1599715 or dx_id = 1572102
    or dx_id = 1676934 or dx_id = 1362357
    or dx_id = 1491580 or dx_id = 1521317
    or dx_id = 1253052 or dx_id = 1253036
    or dx_id = 1225959 or dx_id = 1622577
    or dx_id = 1655829 or dx_id = 1508988
    or dx_id = 1491307 or dx_id = 1362396
    union all 
    (
        SELECT *, 'pac' as glauc_category, 2 as glauc_priority
        FROM `{project_id}.{dataset_id}.{table_id}` 
        Where 
        # pac glaucoma icd codes sophia provided
        lower(icd10) like '%h40.2%' or lower(icd10) like '%h40.03%' 
        or lower(icd10) like '%h40.06%' or lower(icd9) like '%365.2%' 
        or lower(icd9) like '%365.02%' or lower(icd9) like '%365.06%'  

        # when icd10 is null, I used sophia's list of icd9 to label them
        or lower(icd9) = '365.20' or lower(icd9) = '365.23'

        # add false labeled dx_id code to pac glaucoma category
        or dx_id = 1551640 or dx_id = 11511699
        or dx_id = 1527826 or dx_id = 1652999 
        or dx_id = 1535830 or dx_id = 1622411
        or dx_id = 1595677 or dx_id = 1575027 
    )
    union all 
    (
        SELECT *, 'poag' as glauc_category, 3 as glauc_priority
        FROM `{project_id}.{dataset_id}.{table_id}` 
        Where 
        # poag glaucoma icd codes sophia provided
        lower(icd10) like '%h40.10%' or lower(icd10) like '%h40.11%'
        or lower(icd10) like '%h40.12%' or lower(icd10) like '%h40.15%' 
        or lower(icd9) like '%365.10%' or lower(icd9) like '%365.11%'
        or lower(icd9) like '%365.12%' or lower(icd9) like '%365.15%'
        or lower(icd9) like '%365.89%' or lower(icd9) like '%365.9%'

        # when icd10 is null, I used sophia's list of icd9 to label them
        or lower(icd9) = '365.11' or lower(icd9) = '365.12'

        # add false labeled dx_id code to poag glaucoma category
        or dx_id = 1296275 or dx_id = 2216984
        or dx_id = 1185264 or dx_id = 1178801
        or dx_id = 1178803 or dx_id = 1178800
        or dx_id = 1178802 or dx_id = 1185266
        or dx_id = 1296765 or dx_id = 1079391
        or dx_id = 1079392 or dx_id = 1301520
        or dx_id = 1079390 or dx_id = 1079389
        or dx_id = 1301519 or dx_id = 1162769
        or dx_id = 1141462 or dx_id = 1357903
        or dx_id = 1506096 or dx_id = 77692
        or dx_id = 1357455 or dx_id = 1363859
        or dx_id = 1613451 or dx_id = 1819717
        or dx_id = 1545423 or dx_id = 2112160
        or dx_id = 1153812 or dx_id = 1358261
        or dx_id = 1604822 or dx_id = 1650841
        or dx_id = 1819661 or dx_id = 1645340
        or dx_id = 1156721 or dx_id = 1356344
        or dx_id = 1823422 or dx_id = 1725100
        or dx_id = 1730823 or dx_id = 1394901
        or dx_id = 1736879 or dx_id = 1143826
        or dx_id = 1659205 or dx_id = 1641479
        or dx_id = 1568295 or dx_id = 1651931
        or dx_id = 2111398 or dx_id = 163221
        or dx_id = 1125116 or dx_id = 78290
        or dx_id = 159103 or dx_id = 4478
        or dx_id = 1125117 or dx_id = 1215875
        or dx_id = 1162769 or dx_id = 1141462
        or dx_id = 1178801 or dx_id = 1721961
        or dx_id = 1178799 or dx_id = 1651931
        or dx_id = 1725100
    )
    union all 
    (
        SELECT *, 'poag_sus' as glauc_category, 4 as glauc_priority
        FROM `{project_id}.{dataset_id}.{table_id}` 
        Where 

        # poag sus glaucoma icd codes sophia provided
        lower(icd10) like '%h40.00%' or lower(icd10) like '%h40.01%' 
        or lower(icd10) like '%h40.02%' or lower(icd10) like '%h40.05%' 
        or lower(icd9) like '%365.00%' or lower(icd9) like '%365.01%'
        or lower(icd9) like '%365.04%' or lower(icd9) like '%365.05%'

        # when icd10 is null, I used sophia's list of icd9 to label them
        or lower(icd9) = '365.0' 
    )
    order by icd10""".format_map({'project_id': project_id,
                    'dataset_id': dataset_id,
                    'table_id': table_id})
    query_job =client.query(query)
    df=query_job.to_dataframe()
    df.columns = map(str.lower, df.columns)
    df.to_csv(outputfolder+"/glauc_codes.csv", index=None)
    saved_table_id = outputprojectname+".glauc_codes"
    df.to_gbq(saved_table_id, project_id = 'som-nero-phi-sywang-starr', if_exists = 'replace')
    return df


#sql table for non-glauc_codes
def sqlpull_non_glauc_codes(outputfolder, 
                            outputprojectname, 
                            project_id = 'som-nero-phi-naras-ric', 
                            dataset_id = 'Ophthalmology_tables', 
                            table_id = 'sf_oph_dx_dm'): 
    
    """Fetch non glaucoma codes data from bigquery table.
    
    Fetch data from bigquery table stored in gcloud
    and save it as a pandas dataframe.
    
    Args:
        outputfolder: A local directory you choose to store the csv file generated by this function.
        outputprojectname: A project id you created on bigquery that stores the table generated by this function.
        project_id: A id that tells which project the table is stored under.
        dataset_id: A id that tells which dataset the table is stored under that project.
        table_id: A id that assigned to that table (a.k.a table's file name).
    
    Examples:
        sqlpull_non_glauc_codes(outputfolder, outputprojectname)
    
    Returns:
        A dataframe converted from bigquery table.
    """
    
    project_id = project_id
    dataset_id = dataset_id
    query="""
    SELECT *, 'no glaucoma diagnosis' as glauc_category, 5 as glauc_priority
    FROM `{project_id}.{dataset_id}.{table_id}` 
    where dx_id not in (SELECT dx_id FROM `som-nero-phi-sywang-starr.youchen_data_folder.glauc_codes`) 
    """.format_map({'project_id': project_id,
                    'dataset_id': dataset_id,
                    'table_id': table_id})
    query_job =client.query(query)
    df=query_job.to_dataframe()
    df.columns = map(str.lower, df.columns)
    df.to_csv(outputfolder+"/non_glauc_codes.csv", index=None)
    saved_table_id = outputprojectname+".non_glauc_codes"
    df.to_gbq(saved_table_id, project_id = 'som-nero-phi-sywang-starr', if_exists = 'replace')
    return df


#sql table for sf_oph_enc_exam_02_23_iop_cct
def sqlpull_sf_oph_enc_exam_02_23_iop_cct(outputfolder, 
                            outputprojectname):
    """Fetch patients' IOP, CCT, and other feautures from bigquery table.
    
    Fetch data from bigquery table stored in gcloud
    and save it as a pandas dataframe.
    
    Args:
        outputfolder: A local directory you choose to store the csv file generated by this function.
        outputprojectname: A project id you created on bigquery that stores the table generated by this function.
    
    Examples:
        sqlpull_sf_oph_enc_exam_02_23_iop_cct(outputfolder, outputprojectname)
    
    Returns:
        A dataframe converted from bigquery table.
    """
    
    query="""
    SELECT 	pat_enc_csn_id, pat_mrn, pat_id, pat_sex,date_diff(cur_value_datetime, birth_Date, YEAR) as age, ETHNIC, race, line, oph_ta_method, 
        concept_name,iop, oph_ta_time, cct_od, cct_os, cct_date,cur_value_datetime, contact_date, element_value_id	
    FROM 
    (
        SELECT  *
        FROM 
        (
            SELECT *
            FROM 
            (
                SELECT pat_mrn, pat_id, pat_enc_csn_id, line, smrtdta_elem_value as oph_ta_method
                FROM `som-nero-phi-naras-ric.Ophthalmology_tables.sf_oph_enc_exam_02_23` 
                where concept_name like '%OPHTH TA METHOD%' and smrtdta_elem_value is not null 
            ) a
            join 
            (
                SELECT pat_enc_csn_id, line, concept_name, safe_cast(smrtdta_elem_value as int) as iop
                FROM `som-nero-phi-naras-ric.Ophthalmology_tables.sf_oph_enc_exam_02_23` 
                where concept_name like '%IOP%' and smrtdta_elem_value is not null
            ) b
            using (pat_enc_csn_id,line)
            join
            (
                SELECT pat_enc_csn_id, line, smrtdta_elem_value as oph_ta_time
                FROM `som-nero-phi-naras-ric.Ophthalmology_tables.sf_oph_enc_exam_02_23` 
                where concept_name like '%OPHTH TA TIME%' and smrtdta_elem_value is not null
            )
            using (pat_enc_csn_id,line)
            join
            (
                SELECT pat_enc_csn_id, smrtdta_elem_value as cct_od
                FROM `som-nero-phi-naras-ric.Ophthalmology_tables.sf_oph_enc_exam_02_23` 
                where concept_id = 'EPIC#OPH601' and smrtdta_elem_value is not null
            )
            using (pat_enc_csn_id)
            join
            (
                SELECT pat_enc_csn_id, smrtdta_elem_value as cct_os
                FROM `som-nero-phi-naras-ric.Ophthalmology_tables.sf_oph_enc_exam_02_23` 
                where concept_id = 'EPIC#OPH602' and smrtdta_elem_value is not null
            )
            using (pat_enc_csn_id)
            join
            (
                SELECT pat_enc_csn_id, smrtdta_elem_value as cct_date, cur_value_datetime, contact_date, element_value_id
                FROM `som-nero-phi-naras-ric.Ophthalmology_tables.sf_oph_enc_exam_02_23` 
                where concept_id = 'EPIC#OPH603' and smrtdta_elem_value is not null
            )
            using (pat_enc_csn_id)
            where pat_enc_csn_id in 
            (
                #find encounter id that has one tonopen and one applanation but it doesn't mean in this encounter it doesn't contain other ways of measurement
                SELECT pat_enc_csn_id
                FROM 
                (
                    SELECT pat_mrn,pat_id, pat_enc_csn_id,sum(if(lower(smrtdta_elem_value) ='tonopen', 1, 0))as tonopen_sum, sum(if(lower(smrtdta_elem_value) = 'applanation', 1, 0)) as applanation_sum
                    FROM `som-nero-phi-naras-ric.Ophthalmology_tables.sf_oph_enc_exam_02_23` 
                    where concept_name like '%OPHTH TA METHOD%' 
                    group by pat_mrn,pat_id, pat_enc_csn_id
                ) 
                where tonopen_sum=1 and applanation_sum=1
            )
        )
        join
        (
            SELECT distinct pat_id,pat_sex, birth_Date, ETHNIC, race 
            FROM `som-nero-phi-naras-ric.Ophthalmology_tables.sf_oph_patient_02_23`  
        )
        using (pat_id)
    )
    WHERE lower(oph_ta_method) like '%applanation%' or  lower(oph_ta_method) like '%tonopen%' 
    ORDER BY pat_id, pat_enc_csn_id, concept_name,cur_value_datetime asc 

    """
    query_job =client.query(query)
    df=query_job.to_dataframe()
    df.columns = map(str.lower, df.columns)
    df.to_csv(outputfolder+"/sf_oph_enc_exam_02_23_iop_cct.csv", index=None)
    saved_table_id = outputprojectname+".sf_oph_enc_exam_02_23_iop_cct"
    df.to_gbq(saved_table_id, project_id = 'som-nero-phi-sywang-starr', if_exists = 'replace')
    return df



#sql table for left_iop_all_cct
def sqlpull_left_iop_all_cct(outputfolder, 
                            outputprojectname, 
                            project_id = 'som-nero-phi-sywang-starr', 
                            dataset_id = 'youchen_data_folder', 
                            table_id = 'sf_oph_enc_exam_02_23_iop_cct'): 
    
    """Fetch patients' left eye IOP, CCT, and other feautures from bigquery table.
    
    Fetch data from bigquery table stored in gcloud
    and save it as a pandas dataframe.
    
    Args:
        outputfolder: A local directory you choose to store the csv file generated by this function.
        outputprojectname: A project id you created on bigquery that stores the table generated by this function.
    
    Examples:
        sqlpull_left_iop_all_cct(outputfolder, outputprojectname)
    
    Returns:
        A dataframe converted from bigquery table.
    """
    
    project_id = project_id
    dataset_id = dataset_id
    query="""
    SELECT pat_enc_csn_id,	pat_mrn,pat_id,'left' as eye, t1.pat_sex,t1.age, t1.ETHNIC, t1.race,t1.contact_date, t1.cct_os, t1.cct_date, left_tonopen_iop,	left_applanation_iop, left_tonopen_oph_ta_time, left_applanation_oph_ta_time, 
    FROM
    (
        SELECT pat_enc_csn_id, pat_mrn,pat_id,pat_sex, age, ETHNIC, race,contact_date,cct_os, cct_date, iop as left_tonopen_iop, oph_ta_time as left_tonopen_oph_ta_time
        FROM `{project_id}.{dataset_id}.{table_id}` 
        where lower(concept_name) like '%left%' and lower(oph_ta_method) like "%tonopen%" 
    ) t1
    inner join
    (
        SELECT pat_enc_csn_id, pat_mrn,	pat_id,pat_sex, age, ETHNIC, race,contact_date, cct_os,cct_date,iop as left_applanation_iop,oph_ta_time as left_applanation_oph_ta_time
        FROM `{project_id}.{dataset_id}.{table_id}`
        where lower(concept_name) like '%left%' and lower(oph_ta_method) like "%applanation%" 
    ) t2
    using (pat_enc_csn_id, pat_mrn, pat_id)
    """.format_map({'project_id': project_id,
                    'dataset_id': dataset_id,
                    'table_id': table_id})
    query_job =client.query(query)
    df=query_job.to_dataframe()
    df.columns = map(str.lower, df.columns)
    df.to_csv(outputfolder+"/left_iop_all_cct.csv", index=None)
    saved_table_id = outputprojectname+".left_iop_all_cct"
    df.to_gbq(saved_table_id, project_id = 'som-nero-phi-sywang-starr', if_exists = 'replace')
    return df


#sql table for right_iop_all_cct
def sqlpull_right_iop_all_cct(outputfolder, 
                            outputprojectname, 
                            project_id = 'som-nero-phi-sywang-starr', 
                            dataset_id = 'youchen_data_folder', 
                            table_id = 'sf_oph_enc_exam_02_23_iop_cct'): 
    
    """Fetch patients' right eye IOP, CCT, and other feautures from bigquery table.
    
    Fetch data from bigquery table stored in gcloud
    and save it as a pandas dataframe.
    
    Args:
        outputfolder: A local directory you choose to store the csv file generated by this function.
        outputprojectname: A project id you created on bigquery that stores the table generated by this function.
    
    Examples:
        sqlpull_right_iop_all_cct(outputfolder, outputprojectname)
    
    Returns:
        A dataframe converted from bigquery table.
    """
    
    project_id = project_id
    dataset_id = dataset_id
    query="""
    SELECT pat_enc_csn_id,	pat_mrn,pat_id,'right' as eye, t1.pat_sex,t1.age, t1.ETHNIC, t1.race,t1.contact_date, t1.cct_od, t1.cct_date, right_tonopen_iop,	right_applanation_iop, right_tonopen_oph_ta_time, right_applanation_oph_ta_time
    FROM
    (
        SELECT pat_enc_csn_id, pat_mrn,pat_id,pat_sex,age, ETHNIC, race, contact_date, cct_od, cct_date, iop as right_tonopen_iop, oph_ta_time as right_tonopen_oph_ta_time
        FROM `{project_id}.{dataset_id}.{table_id}` 
        where lower(concept_name) like '%right%' and lower(oph_ta_method) like "%tonopen%" 
    ) t1
    inner join
    (
        SELECT pat_enc_csn_id, pat_mrn,	pat_id,pat_sex,age, ETHNIC, race, contact_date, cct_od, cct_date, iop as right_applanation_iop,oph_ta_time as right_applanation_oph_ta_time
        FROM `{project_id}.{dataset_id}.{table_id}` 
        where lower(concept_name) like '%right%' and lower(oph_ta_method) like "%applanation%" 
    ) t2
    using (pat_enc_csn_id, pat_mrn, pat_id)
    """.format_map({'project_id': project_id,
                    'dataset_id': dataset_id,
                    'table_id': table_id})
    query_job =client.query(query)
    df=query_job.to_dataframe()
    df.columns = map(str.lower, df.columns)
    df.to_csv(outputfolder+"/right_iop_all_cct.csv", index=None)
    saved_table_id = outputprojectname+".right_iop_all_cct"
    df.to_gbq(saved_table_id, project_id = 'som-nero-phi-sywang-starr', if_exists = 'replace')
    return df


#sql table for union_left_right_all_cct
def sqlpull_union_left_right_all_cct(outputfolder, 
                            outputprojectname): 

    """Fetch patients' both eyes IOP, CCT, and other feautures from bigquery table.
    
    Fetch data from bigquery table stored in gcloud
    and save it as a pandas dataframe.
    
    Args:
        outputfolder: A local directory you choose to store the csv file generated by this function.
        outputprojectname: A project id you created on bigquery that stores the table generated by this function.
    
    Examples:
        sqlpull_union_left_right_all_cct(outputfolder, outputprojectname)
    
    Returns:
        A dataframe converted from bigquery table.
    """
        
    query="""
    select pat_enc_csn_id,pat_mrn,pat_id,eye,pat_sex,age,ETHNIC as ethnic, race,contact_date, safe_cast(cct_os as int) as cct,cct_date,left_tonopen_iop as tonopen_iop,left_applanation_iop as applanation_iop, left_tonopen_oph_ta_time as tonopen_oph_ta_time,left_applanation_oph_ta_time as applanation_oph_ta_time
    from `som-nero-phi-sywang-starr.youchen_data_folder.left_iop_all_cct`
    union all
    (
        select pat_enc_csn_id,pat_mrn,pat_id,eye,pat_sex,age,ETHNIC,race,contact_date, safe_cast(cct_od as int) as cct,cct_date,right_tonopen_iop as tonopen_iop,right_applanation_iop as applanation_iop, right_tonopen_oph_ta_time as tonopen_oph_ta_time,right_applanation_oph_ta_time as applanation_oph_ta_time
        from `som-nero-phi-sywang-starr.youchen_data_folder.right_iop_all_cct`
    )
    """
    query_job =client.query(query)
    df=query_job.to_dataframe()
    df.columns = map(str.lower, df.columns)
    df.to_csv(outputfolder+"/union_left_right_all_cct.csv", index=None)
    saved_table_id = outputprojectname+".union_left_right_all_cct"
    df.to_gbq(saved_table_id, project_id = 'som-nero-phi-sywang-starr', if_exists = 'replace')
    return df


#sql table for sf_oph_dx_glauc_category_labeled
def sqlpull_sf_oph_dx_glauc_category_labeled(outputfolder, 
                            outputprojectname): 

    """Fetch patients' glaucoma diagnosis top priority from bigquery table.
    
    Fetch data from bigquery table stored in gcloud
    and save it as a pandas dataframe.
    
    Args:
        outputfolder: A local directory you choose to store the csv file generated by this function.
        outputprojectname: A project id you created on bigquery that stores the table generated by this function.
    
    Examples:
        sqlpull_sf_oph_dx_glauc_category_labeled(outputfolder, outputprojectname)
    
    Returns:
        A dataframe converted from bigquery table.
    """
    
    query="""
    SELECT pat_enc_csn_id,	pat_mrn,	pat_id,	contact_date, min(glauc_priority) over (partition by pat_enc_csn_id) as	top_priority
    FROM `som-nero-phi-naras-ric.Ophthalmology_tables.sf_oph_enc_dx_02_23` 
    join
    (
        SELECT *  
        FROM `som-nero-phi-sywang-starr.youchen_data_folder.glauc_codes`
        union all
        (
            SELECT *  
            FROM `som-nero-phi-sywang-starr.youchen_data_folder.non_glauc_codes`
        )
    )
    using (dx_id, dx_name)
    order by pat_id, contact_date, glauc_priority
    """
    query_job =client.query(query)
    df=query_job.to_dataframe()
    df.columns = map(str.lower, df.columns)
    df.to_csv(outputfolder+"/sf_oph_dx_glauc_category_labeled.csv", index=None)
    saved_table_id = outputprojectname+".sf_oph_dx_glauc_category_labeled"
    df.to_gbq(saved_table_id, project_id = 'som-nero-phi-sywang-starr', if_exists = 'replace')
    return df


def top_priority_clean(df,outputfolder, outputprojectname):
    
    """Rearrange patient glaucoma diagnosis priorities
    
    Rearrange patient glaucoma diagnosis priorities 
    and save it as a pandas dataframe.
    
    Args:
        df: A pandas dataframe.
        outputfolder: A local directory you choose to store the csv file generated by this function.
        outputprojectname: A project id you created on bigquery that stores the table generated by this function.
        
    Returns:
        A pandas dataframe.
    """
    
    df_cp = df.copy()
    cumulate = 0
    max_val = 100
    for pat_id, pat_ct in dict(df_cp.groupby('pat_id').size()).items():
        local_min=max_val
        for i in range(cumulate, cumulate + pat_ct):
            if df_cp.iloc[i].pat_id == pat_id:  
                local_min = min(local_min,df_cp.iloc[i]['top_priority'])
                df_cp.at[i,'top_priority_correct'] = local_min
        cumulate = cumulate + pat_ct
    
    df_cp.top_priority_correct = df_cp.top_priority_correct.astype('int', errors='ignore')    
    df_cp.to_csv(outputfolder+"/sf_oph_dx_glauc_category_labeled_corrected.csv", index=None)
    saved_table_id = outputprojectname+".sf_oph_dx_glauc_category_labeled_corrected"
    df_cp.to_gbq(saved_table_id, project_id = 'som-nero-phi-sywang-starr', if_exists = 'replace')    
    return df_cp


#sql table for sf_oph_enc_exam_02_23_iop_cct_glauc_priority
def sqlpull_sf_oph_enc_exam_02_23_iop_cct_glauc_priority(outputfolder, 
                            outputprojectname): 
    
    """Fetch patients' IOP, CCT, glaucoma diagnosis priority, and other features from bigquery table.
    
    Fetch data from bigquery table stored in gcloud
    and save it as a pandas dataframe.
    
    Args:
        outputfolder: A local directory you choose to store the csv file generated by this function.
        outputprojectname: A project id you created on bigquery that stores the table generated by this function.
    
    Examples:
        sqlpull_sf_oph_enc_exam_02_23_iop_cct_glauc_priority(outputfolder, outputprojectname)
    
    Returns:
        A dataframe converted from bigquery table.
    """
    

    query="""
    SELECT distinct *, CASE top_priority_correct
        WHEN 1 THEN 'glaucsec' 
        WHEN 2 THEN 'pac' 
        WHEN 3 THEN 'poag' 
        WHEN 4 THEN 'poag_sus' 
        WHEN 5 THEN 'no glaucoma diagnosis' END as glauc_category 
    FROM
    (
        SELECT * 
        FROM `som-nero-phi-sywang-starr.youchen_data_folder.union_left_right_all_cct`
        join
        ( 
            SELECT pat_enc_csn_id,	pat_mrn,pat_id,	contact_date,top_priority_correct
            FROM `som-nero-phi-sywang-starr.youchen_data_folder.sf_oph_dx_glauc_category_labeled_corrected` 
            order by pat_id, pat_enc_csn_id, contact_date
        )
        using (pat_enc_csn_id,pat_mrn, pat_id,contact_date)
    )
    where cct is not null and tonopen_iop is not null and applanation_iop is not null 
    order by pat_id, pat_enc_csn_id, contact_date, top_priority_correct
    """
    query_job =client.query(query)
    df=query_job.to_dataframe()
    df.columns = map(str.lower, df.columns)
    df.to_csv(outputfolder+"/sf_oph_enc_exam_02_23_iop_cct_glauc_priority.csv", index=None)
    saved_table_id = outputprojectname+".sf_oph_enc_exam_02_23_iop_cct_glauc_priority"
    df.to_gbq(saved_table_id, project_id = 'som-nero-phi-sywang-starr', if_exists = 'replace')
    return df


#preprocess the patient table (two functions, one to get demographics and the other to get behaviors)

def preprocess_pt_dem(dfpt,
                      outputfolder, 
                      outputprojectname): 
    
    """Preprocess patients' demographics infomation.
    
    Args:
        dfpt: A pandas dataframe.
        outputfolder: A local directory you choose to store the csv file generated by this function.
        outputprojectname: A project id you created on bigquery that stores the table generated by this function.
    
    Returns:
        A pandas dataframe.
    """
    
    #lowercase column names 
    dfpt.columns = map(str.lower, dfpt.columns)
    dfdem=dfpt.copy()
    
    #because the original raceth columns are ridiculously messy, including ethnicity in the race column, we are going to create a new column to combine them 
    #everybody hispanic gets to be hispanic, and everyone else is everyone else 

    dfdem['raceth']=dfdem["race"]
    dfdem["raceth"]=dfdem["raceth"].str.replace(", NON-HISPANIC", "")
    dfdem["raceth"]=dfdem["raceth"].str.replace(", HISPANIC", "HISPANIC")
    dfdem["raceth"]=dfdem["raceth"].str.replace(" OR AFRICAN AMERICAN", "")
    dfdem["raceth"]=dfdem["raceth"].str.replace(" - HISTORICAL CONV", "")
    dfdem["raceth"]=dfdem["raceth"].str.replace(" OR AFRICAN AMERICAN", "")
    dfdem["raceth"]=dfdem["raceth"].str.replace("NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER", "ASIAN")
    dfdem["raceth"]=dfdem["raceth"].str.replace("PACIFIC ISLANDER", "ASIAN")
    dfdem["raceth"]=dfdem["raceth"].str.replace("AMERICAN INDIAN OR ALASKA NATIVE", "OTHER") 
    dfdem["raceth"]=dfdem["raceth"].str.replace("NATIVE AMERICAN", "OTHER") 
    dfdem["raceth"]=dfdem["raceth"].str.replace("RACE AND ETHNICITY UNKNOWN", "REFUSED/UNKNOWN") 
    dfdem["raceth"]=dfdem["raceth"].str.replace("UNKNOWN", "REFUSED/UNKNOWN") 
    dfdem["raceth"]=dfdem["raceth"].fillna("REFUSED/UNKNOWN") 
    dfdem["raceth"]=dfdem["raceth"].str.replace("PATIENT REFUSED", "REFUSED/UNKNOWN") 

    dfdem.loc[dfdem.ethnic=="HISPANIC/LATINO", 'raceth']='HISPANIC'
    
    del dfdem["race"]
    del dfdem["ethnic"]

    
    # dfdem.to_csv(outputfolder+"/sf_oph_enc_exam_02_23_iop_cct_glauc_priority_corrected.csv", index=None)
    # saved_table_id = outputprojectname+".sf_oph_enc_exam_02_23_iop_cct_glauc_priority_corrected"
    # dfdem.to_gbq(saved_table_id, project_id = 'som-nero-phi-sywang-starr', if_exists = 'replace')
    return dfdem 


def leftcount(text):
    '''A  regular expression to detect variations of Left/OS and return their count

       Example:  df["left"]=df["note"].apply(leftcount) 
       
       Author: Dr.Sophia Wang
    '''
    lefteye=re.compile('(?i)(LEFT\s*EYE|\s+OS\s+|,\s+OS|[\s_]+L_*\s+eye\s+|(,\s+LEFT)|(LEFT,)|(_+LEFT_+))') 
    return len(lefteye.findall(text))
def rightcount(text):
    '''A regular expression to detect variations of Right/OD and return their count
    
       Example:  df["right"]=df["note"].apply(rightcount)
       
       Author: Dr.Sophia Wang
    '''
    righteye=re.compile('(?i)(RIGHT\s*EYE|\s+OD\s+|,\s+OD|[\s_]+R_*\s+eye\s+|(,\s+RIGHT)|(RIGHT,)|(_+RIGHT_+))') 
    return len(righteye.findall(text))
def laterality(left, right): 
    '''returns 0 if laterality is right eye, 1 if laterality is left eye
    
       Example: df["laterality"]=df[["left", "right"]].apply(lambda x: laterality(*x), axis=1) 
       
       Author: Dr.Sophia Wang
    '''
    if left>right: 
        return 1
    if right>left: 
        return 0  
    

def logmarconversion(va): 
    #takes a string input in the form of "20/20", or "cf" "hm" etc and spits out a logmar 
    if len(re.findall('(?i)cf', va))>0: 
        logmarva=-np.log10(.0025)
    elif len(re.findall('(?i)hm', va))>0: 
        logmarva=-np.log10(.002)
    elif len(re.findall('(?i)nlp', va))>0:
        logmarva=-np.log10(0.0013)
    elif len(re.findall('(?i)lp', va))>0: 
        logmarva=-np.log10(0.0016)
    elif len(re.findall('(?i)20/1600', va))>0: 
        logmarva=-np.log10(20/1600)
    elif len(re.findall('(?i)20/1250', va))>0: 
        logmarva=-np.log10(20/1250)
    elif len(re.findall('(?i)20/1000', va))>0: 
        logmarva=-np.log10(20/1000)
    elif len(re.findall('(?i)20/800', va))>0: 
        logmarva=-np.log10(20/800)
    elif len(re.findall('(?i)20/650', va))>0: 
        logmarva=-np.log10(20/650)
    elif len(re.findall('(?i)20/500', va))>0: 
        logmarva=-np.log10(20/500)
    elif len(re.findall('(?i)20/400', va))>0: 
        logmarva=-np.log10(20/400)
    elif len(re.findall('(?i)20/350', va))>0: 
        logmarva=-np.log10(20/350)
    elif len(re.findall('(?i)20/300', va))>0: 
        logmarva=-np.log10(20/300)
    elif len(re.findall('(?i)20/250', va))>0: 
        logmarva=-np.log10(20/250)
    elif len(re.findall('(?i)20/225', va))>0: 
        logmarva=-np.log10(20/225)
    elif len(re.findall('(?i)20/200', va))>0: 
        logmarva=-np.log10(20/200)
    elif len(re.findall('(?i)20/160', va))>0: 
        logmarva=-np.log10(20/160)
    elif len(re.findall('(?i)20/150', va))>0: 
        logmarva=-np.log10(20/150)
    elif len(re.findall('(?i)20/125', va))>0: 
        logmarva=-np.log10(20/125)
    elif len(re.findall('(?i)20/120', va))>0: 
        logmarva=-np.log10(20/120)
    elif len(re.findall('(?i)20/100', va))>0: 
        logmarva=-np.log10(20/100)
    elif len(re.findall('(?i)20/80', va))>0: 
        logmarva=-np.log10(20/80)
    elif len(re.findall('(?i)20/70', va))>0: 
        logmarva=-np.log10(20/70)
    elif len(re.findall('(?i)20/63', va))>0: 
        logmarva=-np.log10(20/63)
    elif len(re.findall('(?i)20/60', va))>0: 
        logmarva=-np.log10(20/60)
    elif len(re.findall('(?i)20/50', va))>0: 
        logmarva=-np.log10(20/50)
    elif len(re.findall('(?i)20/40', va))>0: 
        logmarva=-np.log10(20/40)
    elif len(re.findall('(?i)20/32', va))>0: 
        logmarva=-np.log10(20/32)
    elif len(re.findall('(?i)20/30', va))>0: 
        logmarva=-np.log10(20/30)
    elif len(re.findall('(?i)20/25', va))>0: 
        logmarva=-np.log10(20/25)
    elif len(re.findall('(?i)20/20', va))>0: 
        logmarva=-np.log10(20/20)
    elif len(re.findall('(?i)20/16', va))>0: 
        logmarva=-np.log10(20/16)
    elif len(re.findall('(?i)20/15', va))>0: 
        logmarva=-np.log10(20/15)
    elif len(re.findall('(?i)20/10', va))>0: 
        logmarva=-np.log10(20/10)
        
    else: logmarva=np.nan 
    return logmarva 


def calc_spheqv(mrxsph, mrxcyl): 
    spheqv = mrxsph + 0.5*mrxcyl 
    return spheqv


def preprocess_exam_rename_concepts(concept_id): 
    return exam_dict[concept_id]


def preprocess_visual_acuity(outputfolder, outputprojectname):
    project_id = 'som-nero-phi-sywang-starr'
    dataset_id = 'youchen_data_folder'
    table_id = 'visual_acuity_glauc_surg_cohort'

    df = sqlpull(client,project_id,dataset_id, table_id=table_id)
    dfexam = df[df["concept_id"].isin(list(exam_dict.keys()))][["concept_id", "concept_name", "pat_mrn", 
                                                                            "pat_enc_csn_id", "contact_date", "start_date",
                                                                            "contact_date_status", "va","laterality"]]
    dfexam["name"]=dfexam["concept_id"].apply(preprocess_exam_rename_concepts)
    dfexam.va = dfexam.va.astype(str)
    dfexam.contact_date = dfexam.contact_date.astype(str)
    dfexam.va = dfexam.va.apply(lambda x: logmarconversion(x))
    df_result = dfexam.groupby(['pat_enc_csn_id','pat_mrn','contact_date', 'contact_date_status', 'laterality']).agg({'va': 'min'}).reset_index()
    df_result.dropna(inplace=True)
    
    df_result.to_csv(outputfolder+"/visual_acuity_cleaned.csv", index=None)
    saved_table_id = outputprojectname+".visual_acuity_cleaned"
    df_result.to_gbq(saved_table_id, project_id = 'som-nero-phi-sywang-starr', if_exists = 'replace')
    return df_result

def pyarrow_dtype_format_correction(df):
    """Correct data frame column data format before uploading dataframe to bigquery
    
    Args:
        df: A Pandas dataframe you trying to correct.
    
    Returns:
        A dataframe.
    """
    for col in df.columns:
            weird = (df[[col]].applymap(type) != df[[col]].iloc[0].apply(type)).any(axis=1)
            if len(df[weird]) > 0:
                print(col)
                df[col] = df[col].astype(str)

            if df[col].dtype == list:
                df[col] = df[col].astype(str)
    return df

# write a function which extracts the laterality of a medication from a patient sig (instructions) 
def eyemedlat(sig, route): 
    
    lefteye=re.compile('(?i)(left|\s+OS)') 
    leftcount= len(lefteye.findall(sig))
    righteye=re.compile('(?i)(right|\s+OD)') 
    rightcount= len(righteye.findall(sig))
    botheyes=re.compile('(?i)(both|\s+OU)')
    bothcount=len(botheyes.findall(sig))
    if bothcount>0: 
        return 2
    if rightcount>leftcount: 
        return 0
    if leftcount>rightcount:
        return 1
    if leftcount==rightcount & leftcount>0: 
        return 2
    if leftcount ==0 & rightcount ==0 & bothcount ==0: 
        #assume bilateral if not specified, conservative assumption; oral meds bilateral also
        return 2 
    else: 
        return None
    
    
def eyemed_laterality_preprocessing(outputfolder,outputprojectname):
    project_id = 'som-nero-phi-sywang-starr'
    dataset_id = 'youchen_data_folder'
    table_id = 'glauc_medication_no_ears'
    eyemeds = sqlpull(client,project_id,dataset_id, table_id=table_id)
    
    #drop the 9 rows where the orderdate is inexplicably after the enddate 
    eyemeds=eyemeds[(eyemeds["order_end_time"]>=eyemeds["order_start_time"])|(eyemeds["order_end_time"].isnull())]
    eyemeds = pyarrow_dtype_format_correction(eyemeds)

    #Then apply to data frame (takes forever since it’s a lambda function) 
    eyemeds['medlat']=eyemeds[["instruction", "med_route"]].apply(lambda x: eyemedlat(*x), axis=1)
    eyemeds.to_csv(outputfolder+"/glauc_medication_no_ear_laterality.csv", index=None)
    saved_table_id = outputprojectname+".glauc_medication_no_ear_laterality"
    eyemeds.to_gbq(saved_table_id, project_id = 'som-nero-phi-sywang-starr', if_exists = 'replace')
    return eyemeds


def fupdated(grp):
    ''' 
    a function which collapses rows with consecutive date ranges
    only works if every row has a start and enddate - doesn't handle missing data well 
    '''
    #define a list to collect valid start and end ranges - this is a list of key:value pairs 
    d=[]
    (
        #append a new row the first row if the start date is at least 2 days greater than the last enddate date from previous row,
        #otherwise update last rows's end date with current row's end date.
        grp.reset_index(drop=True)
           .apply(lambda x: d.append({x.orderdate:x.enddate}) #new saved row 
                            if x.name==0 or (x.orderdate-pd.DateOffset(1))>list(d[-1].values())[0] #if its the first row, or orderdate-1 greater than last saved enddate from previous row
                            else (d[-1].update({list(d[-1].keys())[0]:x.enddate}) #update last rows saved enddate with current rows enddate
                                  if x.enddate>list(d[-1].values())[0] #but only if current rows enddate is greater than last saved rows enddate 
                                  else d[-1].update({list(d[-1].keys())[0]:list(d[-1].values())[0]})), #otherwise, don't update 
                  axis=1)
    )
    #reconstruct a df using only valid start and end dates pairs.
    return pd.DataFrame([[list(e.keys())[0],list(e.values())[0]] for e in d], columns=['orderdate','enddate'])


def check_medication(requests, outputfolder, outputprojectname):
    """Check whether patients intake certain categories of medications with given laterality.
    
    Args:
        requests: A list of requests that includes pat_mrns, dates, medlats, medidcustoms.
        outputfolder: A local directory you choose to store the csv file generated by this function.
        outputprojectname: A project id you created on bigquery that stores the table generated by this function.
    
    Note:
        The variable Requests has to follow the following formats:
            1. pat_mrn has to be a string of digits.
            2. pat_id has to be a string.
            3. date has to be YYYY-MM-DD format.
            4. medlat is one of (0,1,2) values. 
            5. medidcustom is medication customized category code
    
    
    Example:
    requests = [{'pat_mrn':'00023655','date':"2020-03-24",'medlat':2,'medidcustom':13},
            {'pat_mrn':'00023655','date':"2018-09-02",'medlat':0,'medidcustom':13},
            {'pat_mrn':'00025312','date':"2019-11-15",'medlat':2,'medidcustom':11},
            {'pat_mrn':'99497828','date':"2012-12-07",'medlat':2,'medidcustom':19},
            {'pat_mrn':'99463689','date':"2017-05-20",'medlat':2,'medidcustom':3}]
    check_medication(requests, outputfolder, outputprojectname)
    
    Returns:
        A dataframe that tells whether patients intake medications in a given medlat and medidcustom or not.
    """
    
    project_id = 'som-nero-phi-sywang-starr'
    dataset_id = 'youchen_data_folder'
    table_id = 'glauc_medication_no_ear_laterality'
    df = sqlpull(client,project_id,dataset_id, table_id=table_id)
    df = df[['pat_mrn','pat_id', 'order_start_time', 
             'order_end_time', 'medlat', 'medidcustom']].sort_values(['pat_mrn','pat_id', 'order_start_time', 
                                                                      'order_end_time', 'medlat', 'medidcustom'])
    df.order_start_time = pd.to_datetime(df.order_start_time).dt.date
    df.order_end_time = pd.to_datetime(df.order_end_time).dt.date
    
    merged_df = None
    
    for request in tqdm(requests):
        pat_mrn = request['pat_mrn']
        medlat = request['medlat']
        medidcustom = request['medidcustom']
        
        #sanity check for valid pat_mrn, medlat, medidcustom in dataset
        if pat_mrn in df.pat_mrn.values:
            if medlat in df[df['pat_mrn']==pat_mrn].medlat.values:
                if medidcustom in df[(df['pat_mrn']==pat_mrn) & (df['medlat']==medlat)].medidcustom.values:
        
                    df_pat = df[df['pat_mrn']==pat_mrn].query(f'(medlat=={medlat}) & (medidcustom=={medidcustom})')
                    input_date = datetime.strptime(request['date'], "%Y-%m-%d").date()
                    df_pat['input_date'] = input_date
                    df_pat['medication_status'] = df_pat.apply(lambda x: True if (x.medlat=={medlat})
                                                                          & (x.medidcustom=={medidcustom})
                                                                          & (input_date >= x.order_start_time) 
                                                                          & (input_date <= x.order_end_time) else False, axis=1)
                    df_pat = df_pat[['pat_mrn','pat_id','order_start_time','input_date','order_end_time','medlat','medidcustom','medication_status']]


                    if merged_df is None:
                        merged_df = df_pat
                    else:
                        merged_df = pd.concat([merged_df, df_pat], ignore_index=True)
                else:
                    print(f'Medidcustom {medidcustom} does not exist in patient {pat_mrn} records with medlat equals {medlat}.')
                    continue
            else: 
                print(f'Medlat {medlat} does not exist in patient {pat_mrn} records.')
                continue
        else: 
            print(f'Pat_mrn {pat_mrn} does not exist in this dataset.')
            continue
            
            
    merged_df.to_csv(outputfolder+"/medication_status.csv", index=None)
    saved_table_id = outputprojectname+".medication_status"
    merged_df.to_gbq(saved_table_id, project_id = 'som-nero-phi-sywang-starr', if_exists = 'replace')
    
    return merged_df


def gddmodelfeatures(text): 
    """Functions for detecting glaucoma implants from free text operative notes.
    
    Args: 
        text: Free text operative notes.
    
    Return: (gddtype, gddmodel)
    
    Example: df["gddtype"], df["gddmodel"]=zip(*df["note"].apply(gddmodelfeatures))
    Author: Dr. Sophia Wang
    
    Comments:
        At the time this was initially developed, Hydrus was not yet on the market. 
        Since then the function was updated to include this. 
        At the time this was initially developed, iStent Inject was not yet on the market. 
        Since then the function was updated to include this. 
    """
    if len(re.findall('(?i)Baerveldt', text))>0: 
        gddtype="Baerveldt"
        try: gddmodel=re.findall('(?i)Baerveldt.*?([23]50)', text)[0]
        except: gddmodel=None        
    elif len(re.findall('(?i)Ahmed', text))>0: 
        gddtype="Ahmed"
        try: gddmodel=re.findall('(?i)Ahmed.*?(FP[7|8]|S[2|3])', text)[0]
        except: gddmodel=None
    elif len(re.findall('(?i)Molteno', text))>0: 
        gddtype="Molteno"
        try: gddmodel=re.findall('(?i)Molteno.*?([SDMG][1S])', text)[0]
        except: gddmodel=None
    elif len(re.findall('(?i)iStent', text))>0: 
        gddtype="iStent"
        try: gddmodel=re.findall('(?i)iStent\s*(inject)', text)[0]
        except: gddmodel=None
    elif len(re.findall('(?i)CyPass', text))>0: 
        gddtype="CyPass"
        gddmodel=None
    elif len(re.findall('(?i)Xen', text))>0: 
        gddtype="Xen"
        gddmodel=None
    elif len(re.findall('(?i)(Ex-Press|express(\s.*)shunt)', text))>0: 
        gddtype="Ex-PRESS"
        gddmodel=None
    elif len(re.findall('(?i)Hydrus', text))>0: 
        gddtype = "Hydrus"
        gddmodel=None
    else: 
        gddtype=None
        gddmodel=None
    return (gddtype, gddmodel)


def extract_implant_info_from_ops_notes(outputfolder='data', 
                                        outputprojectname='youchen_data_folder'):
    """Extract glaucoma implants information from operative notes
       and upload the result to bigquery.
    
    Args:
        outputfolder: A local directory you choose to store the csv file generated by this function.
        outputprojectname: A project id you created on bigquery that stores the table generated by this function.
    Return:
        df: A dataframe with implants infomation extracted from operative notes.
    """
    
    project_id = 'som-nero-phi-sywang-starr'
    dataset_id = 'youchen_data_folder'
    table_id = 'sf_oph_note_progress_and_operative'
    
    df = sqlpull(client,project_id,dataset_id, table_id=table_id)
    df["gddtype"], df["gddmodel"]=zip(*df["note"].apply(gddmodelfeatures))
    
    df = pyarrow_dtype_format_correction(df)
    df.to_csv(outputfolder+"/sf_oph_note_progress_and_operative_implants.csv", index=None)
    saved_table_id = outputprojectname+".sf_oph_note_progress_and_operative_implants"
    df.to_gbq(saved_table_id, project_id = 'som-nero-phi-sywang-starr', if_exists = 'replace')
    return df


def sqlpull_glauc_diag_surg_cohort_raceth(outputfolder, outputprojectname):
    """Fetch patients' surgeries and glaucoma diagnosis information from bigquery table
       and preprocess race and ethnic to raceth.
    
    Fetch data from bigquery table stored in gcloud
    and save it as a pandas dataframe.
    
    Args:
        outputfolder: A local directory you choose to store the csv file generated by this function.
        outputprojectname: A project id you created on bigquery that stores the table generated by this function.
    
    Examples:
        sqlpull_glauc_diag_surg_cohort_raceth(outputfolder, outputprojectname)
    
    Returns:
        A dataframe converted from bigquery table.
    """
    project_id = 'som-nero-phi-sywang-starr'
    dataset_id = 'youchen_data_folder'
    table_id = 'glauc_diag_surg_cohort'
    df = sqlpull(client,project_id,dataset_id, table_id=table_id)
    df = preprocess_pt_dem(df, outputfolder, outputprojectname)
    
    df = pyarrow_dtype_format_correction(df)
    df.to_csv(outputfolder+"/glauc_diag_surg_cohort_raceth.csv", index=None)
    saved_table_id = outputprojectname+".glauc_diag_surg_cohort_raceth"
    df.to_gbq(saved_table_id, project_id = 'som-nero-phi-sywang-starr', if_exists = 'replace')
    return df


def download_blob(bucket_name, source_blob_name, destination_file_name, verbose=True):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client(project='som-nero-phi-sywang-starr')

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    
    if verbose:
        print(
            "Downloaded storage object {} from bucket {} to local file {}.".format(
                source_blob_name, bucket_name, destination_file_name
            )
        )
    

def upload_blob(bucket_name, source_file_name, destination_blob_name,verbose=True):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    
    if verbose:
        print(
            f"File {source_file_name} uploaded to {destination_blob_name}."
        )

    
def list_blobs(bucket_name):
    """Lists all the blobs in the bucket."""
    # bucket_name = "your-bucket-name"

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)

    for blob in blobs:
        print(blob.name)

        
def write_pdf(ds, filepath):
    with open(f'{filepath}.pdf', 'wb') as fp:
        fp.write(ds.EncapsulatedDocument)
        
        
def extract_pdf_from_batches(bucket_name = 'stanfordoptimagroup',project_id = 'som-nero-phi-sywang-starr',dataset_id = 'imaging',table_id_list = ['patient_details_batch3'], batch_name='batch3'):
    
    t = []
    for table_id in table_id_list:
        t.append(sqlpull(client, project_id, dataset_id, table_id))
    df_cat = pd.concat(t)
    
    d = []
    for idx, filepath in tqdm(enumerate(df_cat.dicomfilepath)):
        source_blob_name = filepath
        output_file_name = 'dcm'+'/' + ('/').join(str(source_blob_name).split('/')[4:])
        if not os.path.exists(output_file_name):
            download_blob('stanfordoptimagroup',source_blob_name, output_file_name)
        ds = pydicom.dcmread(output_file_name)
        try:
            d.append(
                {
                    'PatientID': ds.PatientID,
                    'PatientSex':  ds.PatientSex,
                    'PatientBirthDate':ds.PatientBirthDate,
                    'InstitutionName':ds.InstitutionName,
                    'StudyDate':ds.StudyDate,
                    'StudyID':ds.StudyID,
                    'StudyDescription':ds.StudyDescription,
                    'SeriesDescription':ds.SeriesDescription,
                    'Modality':ds.Modality,
                    'Manufacturer':ds.Manufacturer,
                    'ManufacturerModelName':ds.ManufacturerModelName,
                    'Laterality':ds.Laterality,
                    'DocumentTitle':ds.DocumentTitle,
                    'DicomFilePath':source_blob_name
                }
            )
            print('PDF exists')
        except:
            print('No PDF')
        os.system(f'rm -rf ./dcm/*')
        
    df_pdf = pd.DataFrame(d)
    df_pdf.to_csv(f'EncapsulatedDocument_{batch_name}.csv')
    df_pdf = pyarrow_dtype_format_correction(df_pdf)
    saved_table_id = 'imaging'+f".EncapsulatedDocument_{batch_name}"
    df_pdf.to_gbq(saved_table_id, project_id = 'som-nero-phi-sywang-starr', if_exists = 'replace') 
    return df_pdf


def extract_image_infos_from_batches(bucket_name = 'stanfordoptimagroup',project_id = 'som-nero-phi-sywang-starr',dataset_id = 'imaging',table_id_list = ['patient_details_batch3'], batch_name='batch3'):
    
    t = []
    fieldnames =['PatientID',
            'PatientSex',
            'PatientBirthDate',
            'InstitutionName',
            'SOPClassUID',
            'StudyID',
            'StudyDate',
            'StudyDescription',
            'SeriesDescription', 
            'ImageLaterality',
            'Laterality',
            'Modality',
            'Manufacturer',
            'ManufacturerModelName',
            'ImageType',
            'PhotometricInterpretation',
            'PixelArray',
            'ArrayShape',
            'DicomFilePath']
    
    for table_id in table_id_list:
        t.append(sqlpull(client, project_id, dataset_id, table_id))
    df_cat = pd.concat(t)
    
    if not os.path.exists(f'image_infos_{batch_name}.csv'):
        with open(f'image_infos_{batch_name}.csv', 'w', newline ='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(fieldnames)
            for idx, filepath in tqdm(enumerate(df_cat.dicomfilepath)):
                source_blob_name = filepath
                output_file_name = 'image_infos'+'/' + ('/').join(str(source_blob_name).split('/')[4:])
                if not os.path.exists(output_file_name):
                    download_blob('stanfordoptimagroup',source_blob_name, output_file_name)
                ds = pydicom.dcmread(output_file_name)
                
                try:
                    ds.pixel_array

                except:
                    print('Float Pixel Data or Double Float Pixel Data must be present in the dataset')
                    continue
                
                
                
                rows = []
                for field in fieldnames:
                    if field in ds:
                        rows.append(ds.data_element(field).value)
                    elif field == 'PixelArray':
                        rows.append(ds.pixel_array)
                    elif field == 'ArrayShape':
                        rows.append(ds.pixel_array.shape)
                    elif field == 'DicomFilePath':
                        rows.append(filepath)
                    else:
                        rows.append('')
                writer.writerow(rows)
                os.system(f'rm -rf ./image_infos/*')
    else:
        with open(f'image_infos_{batch_name}.csv', 'a', newline ='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for idx, filepath in tqdm(enumerate(df_cat.dicomfilepath)):
                source_blob_name = filepath
                output_file_name = 'image_infos'+'/' + ('/').join(str(source_blob_name).split('/')[4:])
                if not os.path.exists(output_file_name):
                    download_blob('stanfordoptimagroup',source_blob_name, output_file_name)
                ds = pydicom.dcmread(output_file_name)
                
                try:
                    ds.pixel_array

                except:
                    print('Float Pixel Data or Double Float Pixel Data must be present in the dataset')
                    continue
            
                rows = []
                for field in fieldnames:
                    if field in ds:
                        rows.append(ds.data_element(field).value)
                    elif field == 'PixelArray':
                        rows.append(ds.pixel_array)
                    elif field == 'ArrayShape':
                        rows.append(ds.pixel_array.shape)
                    elif field == 'DicomFilePath':
                        rows.append(filepath)
                    else:
                        rows.append('')
                writer.writerow(rows)
                os.system(f'rm -rf ./image_infos/*')
    

    df = pd.read_csv(f'image_infos_{batch_name}.csv')
    df = pyarrow_dtype_format_correction(df)
    saved_table_id = 'imaging'+f".image_infos_{batch_name}"
    df.to_gbq(saved_table_id, project_id = 'som-nero-phi-sywang-starr', if_exists = 'replace') 
        
    return df


def rnfl_quadrant_search_range(txts):
    ct = 0 
    for i, s in enumerate(txts):
        if 'diversified' in s.lower():
            start_index = i + 1
        if 'rnfl' in s.lower() and 'circular tomogram' in s.lower():
            ct+=1
            if ct == 2:
                end_index = i
    return start_index, end_index

def rnfl_quadrant_search_range_test(txts):
    ct = 0     
    
    for i, s in enumerate(txts):
        if 'normative' in s.lower():
            start_index = i-1
        if 'diversified' in s.lower():
            start_index = i + 1
        if 'quadrants' in s.lower():
            end_index = i+2
    return start_index, end_index+1

def has_numbers(inputString):
    return bool(re.search(r'\d', inputString))

def rnfl_clock_search_range(txts):
    ct = 0 
    for i, s in enumerate(txts):
        
        if 'rnfl' in s.lower() and 'circular tomogram' in s.lower():
            ct+=1
            if ct == 2:
                start_index = i + 1
        if 'comments' in s.lower():
            end_index = i
    return start_index, end_index

def rnfl_clock_search_range_test(txts):
    ct = 0 
    for i, s in enumerate(txts):
        
        if 'quadrants' in s.lower():
            start_index = i+3
        end_index = len(txts)
    return start_index, end_index

def gcc_search_range(txts):
    for i, s in enumerate(txts):
        if '/' in s:
            start_index = i+1
        if 'ave' in s.lower():
            end_index = i
    return start_index, end_index

def zip_file(output_filename, dir_name):
    shutil.make_archive(output_filename, 'zip', dir_name)
    
def pull_pdfs(df):
    """
    Example:
    %%bigquery macula

    SELECT * FROM `som-nero-phi-sywang-starr.imaging.EncapsulatedDocument_all_3_batches` 
    where lower(DocumentTitle) like "%macula%"
    and cast(PatientID as string) in (select mrn from `som-nero-phi-sywang-starr.xlrp_umich.cohort_xlrp`)
    
    #put this line in next notebook cell
    pull_pdfs(macula)
    """
    for idx, filepath in tqdm(enumerate(df.DicomFilePath)):
        source_blob_name = filepath
        output_file_name = 'dcm'+'/' + ('/').join(str(source_blob_name).split('/')[4:])
        if not os.path.exists(output_file_name):
            download_blob('stanfordoptimagroup',source_blob_name, output_file_name,verbose=False)
        ds = pydicom.dcmread(output_file_name)
        write_pdf(ds, f'pdf/{idx}')

    zip_file('pulled_pdf','pdf/')
    upload_blob('stanfordoptimagroup', 'pulled_pdf.zip', 'pulled_pdf.zip',verbose=False)
    os.system(f'rm -rf ./dcm/*')
    os.system(f'rm -rf ./pdf/*')
    os.system(f'rm pulled_pdf.zip')