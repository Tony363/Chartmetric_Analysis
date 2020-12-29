import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import xgboost as xgb
import random
import plotly.express as px

from scipy import stats 
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,StandardScaler,RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.decomposition import PCA
import itertools

# load in main database of songs and attributes
def load_data():
    df = pd.read_csv("Chartmetric_Sample_Data.csv")
    df.drop('Unnamed: 0',axis=1,inplace=True)
    df.reset_index(inplace=True)
    return df

# set some display options so easier to view all columns at once
def set_view_options(max_cols=50, max_rows=50, max_colwidth=9, dis_width=250):
    pd.options.display.max_columns = max_cols
    pd.options.display.max_rows = max_rows
    pd.set_option('max_colwidth', max_colwidth)
    pd.options.display.width = dis_width
    pd.option_context('mode.use_inf_as_na', True)
    

def rename_columns(df):
    # create four dataframes with values, monthly and weekly features
    # and make columns names consistent
    df_fol = df.iloc[:, 3:8]
    df_pop = df.iloc[:, 10:15]   
    df_pop.columns = df_fol.columns
    df_lis = df.iloc[:, 17:22]    
    df_lis.columns = df_fol.columns
    df_flr = df.iloc[:, 24:29]    
    df_flr.columns = df_fol.columns
    
    # add in artist and timestamp
    df_fol = df_fol.add_prefix('fol_')
    df_fol['artist']  = df['Chartmetric_ID']
    df_fol['timestp'] = df['timestp']
    
    df_pop = df_pop.add_prefix('pop_')
    df_pop['artist']  = df['Chartmetric_ID']
    df_pop['timestp'] = df['timestp.1']
        
    df_lis = df_lis.add_prefix('lis_')
    df_lis['artist']  = df['Chartmetric_ID']
    df_lis['timestp'] = df['timestp.2']
        
    df_flr = df_flr.add_prefix('flr_')
    df_flr['artist']  = df['Chartmetric_ID']
    df_flr['timestp'] = df['timestp.3']

    df_fol = pd.merge(df_fol, df_pop, on=['artist', 'timestp'], how='left')
    df_fol = pd.merge(df_fol, df_lis, on=['artist', 'timestp'], how='left')
    df_fol = pd.merge(df_fol, df_flr, on=['artist', 'timestp'], how='left')
    return df_fol

def get_df_info(df):
    # take an initial look at our data
    print(df.head(),'\n')

    # look at data types for each
    info = df.info()
    print(info,'\n')

    # take a look at data types, and it looks like we have a pretty clean data set!
    # However, I think the 0 popularity scores might throw the model(s) off a bit.
    print("Do we have any nulls?")
    print(f"Looks like we have {df.isnull().sum().sum()} nulls\n")
    
    subject_col = []
    statsdf = []
    # look at basic metric mapping
    for idx,col in enumerate(df.columns):
        if idx % 7 != 0:
            try:
                stats = df.agg({col:['min','max','median','mean','skew']})
                subject_col.append(col)
                statsdf.append(stats.transpose())
            except Exception as e:
                # print(e)
                continue
    statsdf = pd.concat(statsdf,axis=0,ignore_index=True)
    statsdf.set_index([pd.Index(subject_col)],inplace=True)
    return statsdf

# calculate and print more stats from the df
def get_stats(df):
    df.reset_index(inplace=True)
    # print stats for various metrics
    print()
    print(f"There are {df.shape[0]} rows")
    print(f"There are {df['artist'].unique().shape} unique artists")
    print(f"There are {df['pop_value'].unique().shape} unique popularity scores")
    print(f"The mean popularity score is {df['pop_value'].mean()}")
    print(f"There are {df[df['pop_value'] > 55]['pop_value'].count()} songs with a popularity score > 55")
    print(f"There are {df[df['pop_value'] > 75]['pop_value'].count()} songs with a popularity score > 75")
    print(f"Only {(df[df['pop_value'] > 80]['pop_value'].count() / df.shape[0])*100:.2f} % of artists have a popularity score > 80")

def scale_grp(df,pct,time,cols):
    pct.replace([np.inf, -np.inf], np.nan,inplace=True)
    df_std = pd.DataFrame(StandardScaler().fit_transform(pct),columns=['fx_followers','fx_popularity','fx_listeners','fx_ratio'])
    df_std['time_series'] = time
    df_std.set_index('time_series',inplace=True)
    df_norm = pd.DataFrame(MinMaxScaler().fit_transform(pct),columns=['fx_followers','fx_popularity','fx_listeners','fx_ratio'])
    df_norm['time_series'] = time
    df_norm.set_index('time_series',inplace=True)
    print([col for col in df.columns if col in cols])
    print([col for col in df.columns if col not in cols])
    return df.loc[:,[col for col in df.columns if col in cols]],df.loc[:,[col for col in df.columns if col not in cols]],df_std,df_norm

def group_time(df,groupby,cols):
    timeidx = [df.columns.get_loc(col) for col in df.columns if "timestp" in col] 
    time = df.iloc[:,timeidx]
    time = time.loc[:,'timestp']
    df = df.drop(df.columns[timeidx[1:]],axis=1)
    df.reset_index(inplace=True)
    df.fillna(method="ffill",inplace=True)
    df = df.groupby(groupby)[cols].first()
    df['fx_followers'] = df['fol_value'].pct_change()
    df['fx_popularity'] = df['pop_value'].pct_change()
    df['fx_listeners'] = df['lis_value'].pct_change()
    df['fx_ratio'] = df['flr_value'].pct_change()
    # separate artists and do fillna one artist at time
    # if artists don't have prev val fill with bfill
    df.fillna(method="ffill",inplace=True)
    pct = df.drop(cols,axis=1)
    return df,pct,time
   

def artist_diff_metric(df):
    timeidx = [df.columns.get_loc(col) for col in df.columns if "timestp" in col] 
    valueidx = [df.columns.get_loc(col) for col in df.columns if "value" in col or "ratio" in col] 
    df.drop(df.columns[timeidx+valueidx],axis=1,inplace=True)
    df.reset_index(inplace=True)
    artist_unique_row = [df.groupby('artist').groups[artist][0] for artist in df.groupby('artist').groups]
    df = df.loc[artist_unique_row,:].set_index('artist')
    df.drop('index',axis=1,inplace=True)
    return df.fillna(method="ffill")

def Multivariable_Matrix(df,original,col):
    cmatrix = []
    for shift,time in zip(range(df.shape[0]),df.index):
        cseries = df.loc[:,col].shift(-shift)
        smatrix = pd.DataFrame({f'shift_{shift}':cseries.values})
        cmatrix.append(smatrix)
    cmatrix = pd.concat(cmatrix,axis=1)
    cmatrix.drop(cmatrix.index[90:], inplace=True)
    cmatrix.index = original.index[:cmatrix.shape[0]]
    return cmatrix
    
def calc_correlations(df, cutoff=0.5):
    corr = df.corr()
    corr_data = corr[corr > cutoff]
    corr_list = df.corr().unstack().sort_values(kind="quicksort",ascending=False)
    return corr_list.where(corr_list < 1.0),corr_data

# nice way to truncate the column names to display easier
# can be used with various metrics
def describe_cols(df, L=10):
    '''Limit ENTIRE column width (including header)'''
    # get the max col width
    O = pd.get_option("display.max_colwidth")
    # set max col width to be L
    pd.set_option("display.max_colwidth", L)
    describe = df.rename(columns=lambda x: x[:L - 2] + '...' if len(x) > L else x).describe()
    pd.set_option("display.max_colwidth", O) 
    return describe

# get redundant pairs from DataFrame
def get_redundant_pairs(df):
    '''Get diagonal pairs of correlation matrix and all pairs we'll remove 
    (since pair each is doubled in corr matrix)'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            if df[cols[i]].dtype != 'object' and df[cols[j]].dtype != 'object':
                # print("THIS IS NOT AN OBJECT, YO, so you CAN take a corr of it, smarty!")
                pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=10):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    
    print("The top absolute correlations are:")
    print(au_corr[0:n])
    return au_corr[0:n]
    
# plot a scatter plot
def scatter_plot(df, col_x, col_y):
    plt.scatter(df[col_x], df[col_y], alpha=0.2)
    plt.title(f"{col_x} vs {col_y}")
    plt.xlabel(f"{col_x}")
    plt.ylabel(f"{col_y}")
    plt.show()

def plot_scatter_matrix(df, num_rows):
    scatter_matrix(df[:num_rows], alpha=0.2, figsize=(6, 6), diagonal='kde')
    plt.show()

# plot a heatmap of the correlations between features as well as dependent variable
def plot_heatmap(df):
    # note this looks better in jupyter as well
    plt.figure(figsize = (16,6))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=True, )
    plt.show()
  
# plot a confusion matrix
def plot_confusion_matrix(cm, ax, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    """
    font_size = 24
    p = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title,fontsize=font_size)
    
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, fontsize=16)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=16)
   
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if i == 1 and j == 1:
            lbl = "(True Positive)"
        elif i == 0 and j == 0:
            lbl = "(True Negative)"
        elif i == 1 and j == 0:
            lbl = "(False Negative)"
        elif i == 0 and j == 1:
            lbl = "(False Positive)"
        ax.text(j, i, "{:0.2f} \n{}".format(cm[i, j], lbl),
                 horizontalalignment="center", size = font_size,
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    ax.set_ylabel('True',fontsize=font_size)
    ax.set_xlabel('Predicted',fontsize=font_size)
    
# plot polularity scores distribution
def plot_pop_dist(df):
    # set palette
    sns.set_palette('muted')

    # create initial figure
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    sns.distplot(df['pop_value']/100, color='g', label="Popularity").set_title("Distribution of Popularity Scores - Entire Data Set")

    # create x and y axis labels
    plt.xlabel("Popularity")
    plt.ylabel("Density")

    plt.show()

# plot undersampling methodology
def undersample_plot(df):
    # set palette
    sns.set_palette('muted')

    # create initial figure
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    sns.distplot(df['pop_value']/100, color='g', label="Popularity").set_title("Illustration of Undersampling from Data Set")
    
    # create line to shade to the right of
    line = ax.get_lines()[-1]
    x_line, y_line = line.get_data()
    mask = x_line > 0.55
    x_line, y_line = x_line[mask], y_line[mask]
    ax.fill_between(x_line, y1=y_line, alpha=0.5, facecolor='red')

    # get values for and plot first label
    label_x = 0.5
    label_y = 4
    arrow_x = 0.6
    arrow_y = 0.2

    arrow_properties = dict(
        facecolor="black", width=2,
        headwidth=4,connectionstyle='arc3,rad=0')

    plt.annotate(
        "First, sample all popularity value in this range.\n Sample size is n. Cutoff is 0.5.", xy=(arrow_x, arrow_y),
        xytext=(label_x, label_y),
        bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.5),
        arrowprops=arrow_properties)

    # Get values for and plot second label
    label_x = 0.1
    label_y = 3
    arrow_x = 0.2
    arrow_y = 0.2

    arrow_properties = dict(
        facecolor="black", width=2,
        headwidth=4,connectionstyle='arc3,rad=0')

    plt.annotate(
        "Next, randomly sample \n n popularity value in this range", xy=(arrow_x, arrow_y),
        xytext=(label_x, label_y),
        bbox=dict(boxstyle='round,pad=0.5', fc='g', alpha=0.5),
        arrowprops=arrow_properties)

    # plot final word box
    plt.annotate(
        "Therefore, end up with a 50/50 \n split of Popular / Not Popular\n artist", xy=(0.6, 2),
        xytext=(0.62, 2),
        bbox=dict(boxstyle='round,pad=0.5', fc='b', alpha=0.5))

    # create x and y axis labels
    plt.xlabel("Popularity")
    plt.ylabel("Density")

    plt.show()

# plot univariate dists for several independent variables
def plot_univ_dists(df, cutoff):
    popularity_cutoff = cutoff
    print('Mean value for followers feature for Popular artists: {}'.format(df[df['pop_value'] > popularity_cutoff]['fol_value'].mean()))
    print('Mean value for followers feature for Unpopular artists: {}'.format(df[df['pop_value'] < popularity_cutoff]['fol_value'].mean()))
    print('Mean value for listeners feature for Popular artists: {}'.format(df[df['pop_value'] > popularity_cutoff]['lis_value'].mean()))
    print('Mean value for listeners feature for Unpopular artists: {}'.format(df[df['pop_value'] < popularity_cutoff]['lis_value'].mean()))
  
    
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    fig.suptitle('Histograms and Univariate Distributions of Important Features')
    sns.distplot(df[df['pop_value'] < popularity_cutoff]['fol_value'])
    sns.distplot(df[df['pop_value'] > popularity_cutoff]['fol_value'])
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    sns.distplot(df[df['pop_value'] < popularity_cutoff]['lis_value'])
    sns.distplot(df[df['pop_value'] > popularity_cutoff]['lis_value'])
    plt.show()

def validation_plot(y_test, pred_test):
    """
    Parameters
    ----------
    y_test : validation set.
    pred_test : prediction on test.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots()
    ax.scatter(y_test, pred_test, color="b")
    # ax.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'k--',lw=4)
    ax.set_xlabel("preds")
    ax.set_ylabel("y_test")
    ax.set_title("validate on pred")
    plt.show()


# choose cutoff, sample popular data, randomly sample unpopular data, and combine the dfs
def split_sample_combine(df, cutoff, col='fx_popularity', rand=None):
    # split out popular rows above the popularity cutoff
    split_pop_df = df[df[col] >= cutoff].copy()

    # get the leftover rows, the 'unpopular' songs
    df_leftover = df[df[col] < cutoff].copy()

    # what % of the original data do we now have?
    ratio = split_pop_df.shape[0] / df.shape[0]
    
    # what % of leftover rows do we need?
    ratio_leftover = split_pop_df.shape[0] / df_leftover.shape[0]

    # get the exact # of unpopular rows needed, using a random sampler
    unpop_df_leftover, unpop_df_to_add = train_test_split(df_leftover, \
                                                          test_size=ratio_leftover, \
                                                          random_state = rand)
    
    # combine the dataframes to get total rows = split_pop_df * 2
    # ssc stands for "split_sample_combine"
    ssc_df = split_pop_df.append(unpop_df_to_add).reset_index(drop=True)

    # shuffle the df
    ssc_df = ssc_df.sample(frac=1, random_state=rand).reset_index(drop=True)
    
    # add columns relating to popularity
    ssc_df['pop_frac'] = ssc_df[col] / 100
    ssc_df['pop_cat'] = np.where(ssc_df[col] > cutoff, "Popular", "Not_Popular")
    ssc_df['pop_bin'] = np.where(ssc_df[col] > cutoff, 1, 0)
    return ssc_df

# initial linear regression function, and plots
def linear_regression_initial(df,features,Y):
    df = df.copy()
    X = df[features]
    plot_title = (Y) if len(features) > 1 else (Y,features)
    y_col = [Y]
    y = df[y_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = sm.add_constant(X_train)
    # Instantiate OLS model, fit, predict, get errors
    model = sm.OLS(y_train, X_train)
    results = model.fit()
    fitted_vals_train = results.predict(X_train)
    stu_resid = results.resid_pearson
    residuals = results.resid
    y_vals = pd.DataFrame({'residuals':residuals, 'fitted_vals':fitted_vals_train, \
                            'stu_resid': stu_resid})

    # Print the results
    print(results.summary())

    # QQ Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.title(f"QQ Plot {plot_title} of OLS")
    fig = sm.qqplot(stu_resid, line='45', fit=True, ax=ax)
    plt.show()

    # validate pred
    validation_plot(y_train, fitted_vals_train)

    # Residuals Plot
    y_vals.plot(kind='scatter', x='fitted_vals', y='stu_resid')
    plt.title(f"{plot_title} regression")
    plt.show()
    return y_vals

def lin_reg_forcast(df,Y,PARAMETERS = {
        'max_depth':20,
        'min_child_weight': 5,
        'eta':.1,
        'subsample': .7,
        'colsample_bytree': .7,
        'nthread':-1,
        'objective':'reg:squarederror',
        'eval_metric':'rmse'
    }):
    df = df.copy()
    X_cols = df.columns.drop(Y)

    y_col = [Y]

    X = df[X_cols]
    y = df[y_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    dtrain = xgb.DMatrix(X_train, y_train,nthread=-1)
    dtest = xgb.DMatrix(X_test, y_test,nthread=-1)

    bst = xgb.train(
        params=PARAMETERS,
        dtrain=dtrain,
        # num_boost_round=999,
        # evals=[(dtest,"Test")],
        # early_stopping_rounds=10,
        )
    pred_test = bst.predict(dtrain)
    pred_leaf = bst.predict(dtrain,pred_leaf=True)
    print(pred_leaf)
    return pred_test

# Create a basic logistic regression
def basic_logistic_regression(df, cutoff,col='pop_bin', rand=0, sig_only=False):
    df = df.copy()
    X, y = return_X_y_logistic(split_sample_combine(df, cutoff,col, rand=rand))
    X = standardize_X(X)
    X_const = add_constant(X, prepend=True)
    print("X_const\n",X_const)
    print("Y\n",y)
    
    logit_model = Logit(y, X_const).fit(solver='lbfgs',skip_hessian=True,max_iter=20000)
    
    print(logit_model.summary())

    return logit_model

def logistic_regression_with_kfold(df, cutoff=2.682048, rand=0, sig_only=False):
    df = df.copy()
    
    if sig_only == True:
        X, y = return_X_y_logistic_sig_only(split_sample_combine(df, cutoff=cutoff, rand=rand))
        X = standardize_X_sig_only(X)

    else:
        X, y = return_X_y_logistic(split_sample_combine(df, cutoff=cutoff, rand=rand))
        X = standardize_X(X)

    X = X.values
    y = y.values.ravel()

    classifier = LogisticRegression()

    # before kFold
    y_predict = classifier.fit(X, y).predict(X)
    y_true = y
    accuracy_score(y_true, y_predict)
    print(f"accuracy: {accuracy_score(y_true, y_predict)}")
    print(f"precision: {precision_score(y_true, y_predict)}")
    print(f"recall: {recall_score(y_true, y_predict)}")
    print(f"The coefs are: {classifier.fit(X,y).coef_}")

    # with kfold
    kfold = KFold(len(y))

    accuracies = []
    precisions = []
    recalls = []

    for train_index, test_index in kfold:
        model = LogisticRegression()
        model.fit(X[train_index], y[train_index])

        y_predict = model.predict(X[test_index])
        y_true = y[test_index]

        accuracies.append(accuracy_score(y_true, y_predict))
        precisions.append(precision_score(y_true, y_predict))
        recalls.append(recall_score(y_true, y_predict))

    print(f"accuracy: {np.average(accuracies)}")
    print(f"precision: {np.average(precisions)}")
    print(f"recall: {np.average(recalls)}")

def Principal_Comp_Reg(df,features,Y,Standardize=False,n_components=2):
    pca = PCA(n_components=n_components)
    # Separating out the features
    x = df.loc[:,features].values
    # Separating out the target
    y = df.loc[:,[Y]].values
    if Standardize:
        # Standardizing the features
        x = StandardScaler().fit_transform(x)
    x = pca.fit_transform(x)
    principalComponents = np.hstack((x,y))
    principaldf = pd.DataFrame(principalComponents,columns=[f"P{i}" for i in range(n_components)]+[Y])

    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    fig = px.scatter_matrix(
        principalComponents,
        labels=labels,
        dimensions=range(3),
        color=df[Y],
        title="PCA of X+y"
    )
    fig.update_traces(diagonal_visible=False)
    fig.show()

    recons = pca.inverse_transform(x)
    reconsdf = pd.DataFrame(recons,columns=features)

    fig = px.scatter_matrix(
        principalComponents,
        labels=labels,
        dimensions=range(3),
        color=df[Y],
        title="Inverse PCA of X+y"
    )
    fig.update_traces(diagonal_visible=False)
    fig.show()

    return principaldf,pca.explained_variance_ratio_,reconsdf

def Visualize_PCA(df,features,Y,cluster_range=0.2):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)

    colors = ['r', 'g', 'b']

    for feature, color in zip(features,colors):
        ax.scatter(df.loc[(df[feature] > df[Y]-cluster_range) & (df[feature] < df[Y]+cluster_range),'P0']
                , df.loc[(df[feature] > df[Y]-cluster_range) & (df[feature] < df[Y]+cluster_range),'P1']
                , c = color
                , s = 50)
    ax.legend(features)
    ax.grid()

def Kmeans_elbow(X):
    wcss = []
    for i in range(1,11):
        kmeans = KMeans(n_clusters=i,init='k-means++',random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(10,5))
    sns.lineplot(range(1, 11), wcss,marker='o',color='red')
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    # Fitting K-Means to the dataset
    kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(X)
    return y_kmeans,kmeans

def Kmeans_Viz(X,y_kmeans,kmeans):
    # Visualising the clusters
    plt.figure(figsize=(15,7))
    sns.scatterplot(X.iloc[y_kmeans == 0, 0], X.iloc[y_kmeans == 0, 1], color = 'yellow', label = 'Cluster 1',s=50)
    sns.scatterplot(X.iloc[y_kmeans == 1, 0], X.iloc[y_kmeans == 1, 1], color = 'blue', label = 'Cluster 2',s=50)
    sns.scatterplot(X.iloc[y_kmeans == 2, 0], X.iloc[y_kmeans == 2, 1], color = 'green', label = 'Cluster 3',s=50)
    sns.scatterplot(X.iloc[y_kmeans == 3, 0], X.iloc[y_kmeans == 3, 1], color = 'grey', label = 'Cluster 4',s=50)
    sns.scatterplot(X.iloc[y_kmeans == 4, 0], X.iloc[y_kmeans == 4, 1], color = 'orange', label = 'Cluster 5',s=50)
    sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red', 
                    label = 'Centroids',s=300,marker=',')
    plt.grid(False)
    plt.title('Artist Popularity Clustering')
    plt.xlabel('Popularity 0-100')
    plt.ylabel('Followers/Listeners')
    plt.legend()
    plt.show()

# various data standardization and X/y split functions for logisitic reression
# based on the columns you want to standardize and return
def return_X_y_logistic(df):
    df = df.copy()

    # define columns to use for each
    X_cols = ['fx_followers','fx_listeners','fx_ratio','pop_frac']

    # use 1's and 0's for logistic
    y_col = 'pop_bin'

    # split into X and y
    X = df[X_cols]
    y = df[y_col]

    return X, y

def standardize_X(X):  
    X = X.copy()
    
    # standardize only columns not between 0 and 1
    for col in ['fx_followers','fx_listeners','fx_ratio','fx_popularity','pop_frac']:
        new_col_name = col + "_std"
        X[new_col_name] = (X[col] - X[col].mean()) / X[col].std()
        
    X_cols = ['fx_followers_std','fx_popularity_std','fx_listeners_std','fx_ratio_std','pop_frac_std']

    # return the std columns in a dataframe
    X = X[X_cols]
    
    return X


    
if __name__ == "__main__":
    # data clean
    df = load_data()  
    set_view_options(max_cols=50, max_rows=50, max_colwidth=40, dis_width=250)
    duplicated = True in df.columns.duplicated()
    print(f"duplicate columns: {duplicated}\n")
    df = rename_columns(df)
    
    # only time series
    groupby = ['timestp']
    cols = ['fol_value','pop_value','lis_value','flr_value']
    grp,pct,time = group_time(df,groupby,cols)
    grp,cr,cr_std,cr_norm = scale_grp(grp,pct,time,cols)
    
    # sort by ID and time series
    groupby = ['artist','timestp']
    cols = ['fol_value','pop_value','lis_value','flr_value']
    grped,pct,time = group_time(df,groupby,cols)
    grped,cr,cr_std,cr_norm = scale_grp(grped,pct,time,cols)
    
    grp.to_csv("grouped.csv")
    cr.to_csv("change_rate.csv")
    cr_std.to_csv("change_rate_standardized.csv")
    cr_norm.to_csv("change_rate_normalized.csv")
    cr_stdnorm.to_csv("change_rate_standardized.csv")
    
    artist_diff = artist_diff_metric(df)
    artist_diff.to_csv("artist_diff.csv")
    artist_diff.head()
    
    get_df_info(artist_diff)
    cr_std = cr_std.iloc[1:,:]
    get_df_info(cr_std)
    get_df_info(cr_norm)
    cr_stdnorm = cr_stdnorm.iloc[1:,:]
    get_df_info(cr_stdnorm) 
    get_df_info(grp)
    plot_heatmap(grp)
    
    
    artist_PCA,explained_ratio,artist_recons = Principal_Comp_Reg(artist_diff,artist_diff.columns.drop('pop_monthly_diff'),'pop_monthly_diff',Standardize=True)
    Visualize_PCA(artist_PCA,artist_PCA.columns.drop('pop_monthly_diff'),'pop_monthly_diff')
    
    time_PCA,explained_ratio,time_recons = Principal_Comp_Reg(grp,grp.columns.drop('pop_value'),'pop_value',n_components=2)
    
    # if LSTM generate batches by artist
    # Tree based methods generate integer ID for artist
    time_pca_reg = linear_regression_initial(time_PCA,time_PCA.columns.drop('pop_value'),Y='pop_value')
        
    grp_PCA,explained_ratio,grp_recons = Principal_Comp_Reg(grped,grped.columns.drop('pop_value'),'pop_value',n_components=2)
    
    pca_reg = linear_regression_initial(grp_PCA,grp_PCA.columns.drop('pop_value'),Y='pop_value')
    
    y_kmeans,kmeans = Kmeans_elbow(time_PCA)
    Kmeans_Viz(time_PCA,y_kmeans,kmeans)
    
    # Decided Time Series is most prevalent data frame to use
    # Find Highly correlated metrics to list and dataframe
    corr_list,corr_data = calc_correlations(grp)
    corr_list
    
    # index corr_list by correlation greater than 0.5
    plot_index = corr_list[corr_list > 0.5].index
    # plot their pair x y relationship
    for plot in plot_index:
      scatter_plot(grp,plot[0],plot[1])
    
    for plot in plot_index:
        linear_regression_initial(grp,[plot[0]],plot[1])
    describe_cols(grp,40)
    plot_pop_dist(grp)
    undersample_plot(grp)
    plot_univ_dists(grp, 70)
    
    au_corr = get_top_abs_correlations(grp, 25)
    # get unique columns from top abs correlations
    train_cols = np.unique((np.asarray([(index[0],index[1]) for index in au_corr.index])).flatten())
    # index grouby Chartmetric ID and timeseries with the unique top abs correlations
    dtrain = grp[train_cols]
    # plot newly abs correlated heatmap
    plot_heatmap(dtrain)
    
    fitted_pop = linear_regression_initial(dtrain,dtrain.columns.drop('pop_value'),Y='pop_value')
    artist_scores = stats.zscore(fitted_pop['fitted_vals'])
    get_df_info(fitted_pop)
    MM = Multivariable_Matrix(pd.DataFrame(stats.zscore(fitted_pop),columns=fitted_pop.columns),fitted_pop,'fitted_vals')
    pred_over90 = lin_reg_forcast(MM,Y='shift_0')
