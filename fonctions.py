# Import library
import pylab 
import warnings
import numpy as np 
import pandas  as pd 
import seaborn as sns
from statistics import mode
import scipy.stats as stats
import statsmodels.api as sm
from sklearn import set_config 
import matplotlib.pyplot as plt 
from sklearn import linear_model
from IPython.display import Image
from scipy.stats import norm, skew
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
warnings.filterwarnings('ignore')
set_config(display='diagram')
# plt.style.use("fivethirtyeight")
sns.set()


class Fonctions:
    
    def read_original_data(self, path):
        return pd.read_csv(path, sep =';')
    
    def analys_form_dataframe(self, data):
        print('---------------------------------------------------------------------------------------------------------------------------')
        print("------------------------------------------- The dataframe's shape:", data.shape,"------------------------------------------")
        print('---------------------------------------------------------------------------------------------------------------------------')
        print('---------------------------------------------------  Head dataframe  ------------------------------------------------------')
        print('---------------------------------------------------------------------------------------------------------------------------')
        return data.head()
    
    def show_dist(self, x, data, title):
        plt.figure(figsize=(15, 7))

        sns.distplot(data[x], fit=norm)
        (mu, sigma) = norm.fit(data[x])

        plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu,sigma)], loc='best')
        plt.ylabel('Frequency')
        plt.title(title)

    #     fig = plt.figure()
        plt.figure(figsize=(15, 7))
        res = stats.probplot(data[x], plot=plt)
        plt.show()
        print("Skewness : %.2f" % data[x].skew())
        print("Kurtosis : %.2f" % data[x].kurt())
        return            
    
    def comparaison(self):
        comparaison = [(0.79, 0.99, 0.99, 0.52, 0.51, 0.98),
                       (0.79, 0.99, 0.99, 0.52, np.nan, 0.98),
                       (161, 27, 26.92, 246, np.nan, 39), 
                       (0.79, 0.99, 0.99, 0.79, np.nan, 0.98),
                       (0.79, 0.99, 0.99, 0.79, np.nan, 0.98),
                       (161, 27, 26.93, 161, np.nan, 39),
                       (0.81, np.nan, np.nan, np.nan, 0.51, 0.98),
                       (0.82, np.nan,np.nan, np.nan, 0.51, 0.98),
                       (127,np.nan,np.nan, np.nan, 250, 39),
                       (np.nan, 0.99, 0.99, 0.99, np.nan, 0.98),
                       (np.nan, 0.31, 0.22, 0.31, np.nan, 0.15),
                       (np.nan, 279, 297, 279, np.nan, 316)]
        index = ['ridge_val', 'ridge_r2_test', 'ridge_mae',
                 'lasso_val', 'lasso_r2_test', 'lasso_mae',
                 'linear_val', 'linear_r2_test', 'linear_mae',
                 'elastic_val', 'elastic_r2_test', 'elastic_mae']
        columns = ['1_Wind', '4_feat.', '3_feat.',
                   '2_feat.', '1_Generator', '1_Torque']

        comparaison_model = pd.DataFrame(data=comparaison, index = index, columns = columns)

        # 1 feature : windspeed / generator speed / torque
        # 2 features : Powerfactor & rotorspeed
        # 3 features : Powerfactor & torque & rotorspeed
        # 4 features : Speed & powerfactor & torque & rotorspeed
        return comparaison_model

    def delet_features_having_more_then_90_per_cent_miss_values(self, data):
        return data[data.columns[data.isna().sum()/data.shape[0] <0.9]]

    
    def delet_features_having_more_then_80_per_cent_miss_values(self, data):
        return data[data.columns[data.isna().sum()/data.shape[0] <0.8]]

    
    def delet_features_having_more_then_70_per_cent_miss_values(self, data):
        return data[data.columns[data.isna().sum()/data.shape[0] <0.7]]

        
    def delet_features_having_more_then_50_per_cent_miss_values(self, data):
        data = data[data.columns[data.isna().sum()/data.shape[0] <0.5]]
        return data.dropna()
    
    def dataframe_keys(self, data):
        columns = []
        keys = data.keys()
        for k in keys:
            columns.append(k)
        return columns


    def convet_datatime(self,time, data):
        return pd.to_datetime(data[time], infer_datetime_format=True)
    
    def get_month(self, date_time, data):
        data['Month'] = [d.month for d in data[date_time]]
        return data['Month']
    
    def get_year(self, date_time, data):
        data['Year'] = [d.year for d in data[date_time]]        
        return data['Year']
    
    def convert_wind_speed(self, data, Ws):
        data["Speed km/h"] = data[Ws] / 1000
        data["Speed km/h"] = data["Speed km/h"] * 3600
        return data["Speed km/h"]
    
    def show_map_missing_values(self, data):
        plt.figure(figsize=(17,7))
        heatmap = sns.heatmap(data.isna(), cbar=False, cmap="Blues")
        heatmap.set_title('Heatmap of missing values in the dataframe',
        fontdict={'fontsize':18}, pad=16);

    
    def heatmap_corr(self,data, liste):
        plt.figure(figsize=(15, 11))
        mask = np.triu(np.ones_like(data[liste].corr(),dtype=np.bool))
        heatmap = sns.heatmap(data[liste].corr(),
                              vmin=-1,
                              vmax=1,
                              mask=mask,
                              annot=True,
                              cmap="Blues")
        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':25}, pad=12);
    
    
    def heatmap_corr_target_features(self, target, data):
        plt.figure(figsize=(5, 7))
        heatmap = sns.heatmap(data.corr()[[target]].sort_values(by=target, ascending=False)[:30],
                              vmin=-1,
                              vmax=1,
                              annot=True,
                              cmap='Blues')
        heatmap.set_title('Features Correlating with target',
                          fontdict={'fontsize':18},
                          pad=16);     
        
        
    def show_map_missing_values(self, df):
        plt.figure(figsize=(17,7))
        heatmap = sns.heatmap(df.isna(), cbar=False, cmap="Blues")
        heatmap.set_title('Heatmap of missing values in the dataframe',
        fontdict={'fontsize':18}, pad=16);
        
        
    def join_residplot(self, data, x, y):
        #jointplot of target and variable_x, regression
        sns.jointplot(x, y, data, kind='reg', color='b')
        plt.show()
        # draw residplot
        sns.residplot(x, y, data)
        plt.show()
        
    
    def hisplot_windspeed(self, data, speed, categ):
        plt.figure(figsize=(15, 7))
        sns.histplot(x=speed, data=data, hue=categ, kde=True, stat="density")
        plt.xlabel('Speed km/h')
        plt.ylabel('f. Density #Speed km/h')
        plt.title("Histogram density function of Speed km/h")
        plt.xlim(0,90)
        plt.show()
        
    
    def hisplot_active_power(self, data, power, z):   
        plt.figure(figsize=(15, 7))
        sns.histplot(x=power, data=data, hue=z, kde=True, stat="density")
        plt.xlabel('Avtive Power Wind Energy KW')
        plt.ylabel('f. Density #Power Wind Energy KW')
        plt.title("Histogram Histogram density function of Active Power Wind Energy KW")
        plt.xlim(-17,2050)
        plt.show()
    
    def lineplot_month(self, data, date_time, fauture, categ, month):
        plt.figure(figsize=(15, 7))
        sns.lineplot(x= date_time, y=fauture, 
                     hue=categ, 
                     markers=True,
                     lw=1,
                     data=data.query(month))
        
    
    def barplot_active_power(self, categ, power, data):    
        plt.figure(figsize=(15,7))
        sns.barplot(y=categ,x=power,data=data)
        plt.title("Active Power Wind Energy KW by Wind_turbine_name")
        plt.xlabel('Active Power Wind Energy KW')     
        
        
    def barplot_month(self, month, power, data):
        plt.figure(figsize=(15,7))
        sns.barplot(x=month, y=power, data=data)
        plt.title("Active Power Wind Energy KW by Months")
        plt.ylabel('Active Power Wind Energy KW')     
     
    
    def features_selection_and_split_with_columns_to_drop(self, data, col_to_drop, target):
        X = data.drop(columns = col_to_drop, axis=1)
        y = data[target]
        # Split la donnée
        return train_test_split(X, y, test_size=0.20, random_state=5)
    
    
    def features_selection_and_split_with_liste_of_Xfeatures(self, data, liste_of_Xfeatures, target):
        X = data[liste_of_Xfeatures]
        y = data[target]
        # Split la donnée
        return train_test_split(X, y, test_size=0.20, random_state=5)  
    

    def show_split_shape(self,X_train, X_test, y_train, y_test):
        print('Shape X_train set :', X_train.shape)
        print('Shape y_train set :', y_train.shape)
        print('Shape X_test set :', X_test.shape)
        print('Shape y_test set :', y_test.shape)
        
        
    def statmod_function(self, X_train, y_train):
        X_train_constant = sm.add_constant(X_train)
        model_lin_reg = sm.OLS(y_train, X_train_constant)
        results = model_lin_reg.fit()
        print(results.summary())
        
        
    def linear_regression_model(self, X_train, X_test, y_train, y_test, x1, x2, x3, x4):
        model = LinearRegression()
        model.fit(X_train, y_train)        
        print('---------------------------------------------------------------------------------------------------------------------------')
        print('------------------------------------  Score model sur les Train Sets avec Sklearn  ----------------------------------------')
        print('---------------------------------------------------------------------------------------------------------------------------')
        score_model_train = model.score(X_train, y_train)
        print('\t\t\t\t\t\t R² =', score_model_train)
        print('---------------------------------------------------------------------------------------------------------------------------')
        print("-------------------------------------  Cross Validation : Accuracy - MSE : Train ------------------------------------------")
        print('---------------------------------------------------------------------------------------------------------------------------')
        crossValSCORESTrain = cross_val_score(model, X_train, y_train, cv = 6)
        crossValSCORESTrain = pd.DataFrame(crossValSCORESTrain, columns = ['CV Accuracy Train'])
#         print(crossValSCORESTrain)
        crossValMSETrain = cross_val_score(model,X_train, y_train,
                                           scoring='neg_mean_squared_error', cv = 6)
        crossValMSETrain = pd.DataFrame(crossValMSETrain, columns = ['CV MSE Train'])
#         print(crossValMSETrain)
        cv_train = crossValSCORESTrain.join(crossValMSETrain).T
        print(cv_train)
        print('\n')
        print('---------------------------------------------------------------------------------------------------------------------------')
        print('------------------------------------  Score model sur les Test Sets avec Sklearn  -----------------------------------------')
        print('---------------------------------------------------------------------------------------------------------------------------')
        score_model_test = model.score(X_test, y_test)
        print('\t\t\t\t\t\t R² =', score_model_test)
        print('---------------------------------------------------------------------------------------------------------------------------')
        print("-------------------------------------  Cross Validation : Accuracy - MSE : Test -------------------------------------------")
        print('---------------------------------------------------------------------------------------------------------------------------')
        crossValSCORESTest = cross_val_score(model, X_test, y_test, cv = 6)
        crossValSCORESTest = pd.DataFrame(crossValSCORESTest, columns = ['CV Accuracy Test'])
#         print(crossValSCORESTest)
        crossValMSEtest = cross_val_score(model,X_test, y_test,
                                           scoring='neg_mean_squared_error', cv = 6)
        crossValMSEtest = pd.DataFrame(crossValMSEtest, columns = ['CV MSE Test'])      
#         print(crossValMSEtest)
        cv_test = crossValSCORESTest.join(crossValMSEtest).T
        print(cv_test)
        
        y_pred_train = model.predict(X_train)
        residuals_train = y_pred_train - y_train
        
#         plt.figure(figsize=(15, 5))
#         sns.histplot(data=residuals_train, kde=True)
#         plt.xlim(x1,x2)
#         plt.title('Distribution des résidus de model prediction Trains')
#         plt.show()
# #         print(residuals_train.describe())
#         print('\t\t\t\t\t ', f"Moyenne des résidus : {np.mean(residuals_train)}")
#         print('\t\t\t\t\t ', f"Médiane des résidus : {np.median(residuals_train)}")
#         print('\t\t\t\t\t ', f"Mode des résidus : {mode(residuals_train)}")
        
#         # QQ plot : permet de vérifier si la distribution suit une loi normale
#         plt.figure(figsize=(15, 5))
#         stats.probplot(residuals_train, dist="norm", plot=pylab)
#         pylab.show()

        y_pred_test = model.predict(X_test)
        residuals_test = y_pred_test - y_test
        
        plt.figure(figsize=(15, 5))
        sns.histplot(data=residuals_test, kde=True)
        plt.xlim(x3,x4)
        plt.title('Distribution des résidus de model prediction Tests')
        plt.show()
#         print(residuals_test.describe())
        print('\t\t\t\t\t ', f"Moyenne des résidus : {np.mean(residuals_test)}")
        print('\t\t\t\t\t ', f"Médiane des résidus : {np.median(residuals_test)}")
        print('\t\t\t\t\t ', f"Mode des résidus : {mode(residuals_test)}")
        
        # QQ plot : permet de vérifier si la distribution suit une loi normale
        plt.figure(figsize=(15, 5))
        stats.probplot(residuals_test, dist="norm", plot=pylab)
        pylab.show()
        
        return cv_train, cv_test, y_pred_train, y_pred_test
    
    
    def vif_function(self, data, features):
        # variance_inflation_factor parameters : 
        # On sélectionne les features en fonction de la matrice de correlation - finalement cela ne nous aide pas, car
        # elles sont très multicolinéaires => "ajustement manuel"
        print('---------------------------------------------------------------------------------------------------------------------------')
        print("------------------------------------------  Variance Inflation Factor -----------------------------------------------------")
        print('---------------------------------------------------------------------------------------------------------------------------')
        X = data[features]
        vif = pd.DataFrame()
        vif["Features"] = X.columns
        vif["vif_factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
   
        return vif.round()

    def model_evaluate(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return r2, mae
    def show_cv_r2_mae(self,cv_r2, r2,mae):
        cv_r2 = np.mean(cv_r2)
        print("Cross val score: " + str(cv_r2))
        r2, mae = model_evaluate(y_test, y_preds)
        print("R^2 score: " + str(r2))
        print("Mean Absolute Erro: " + str(mae))
    
    def pipeline_preprocessing(self, X):
        # On définit les variables numériques et catégorielles
        categorical_features = X.select_dtypes(include=['object']).columns
        numerical_features = list(X.columns.values)
        
        # Mise en place pipeline pour chaque catégorie de variable
        # Pas de NaN pour les valeurs catégorielles donc pas de SimpleImputer
        # Remplacement des NaN valeurs nuémriques par la médiane car nous avons des outliers non gérés
        # Robuscaler() : moins sensible aux valeurs extrêmes

        pipe_categorical = make_pipeline(OneHotEncoder(handle_unknown='ignore'))
        pipe_numerical  = make_pipeline(SimpleImputer(missing_values=np.nan, strategy="median"), RobustScaler())
        # On transforme les colonnes crées avec les pipeline de preprocessing
        # (pipe_categorical,categorical_features)

        preprocessing = make_column_transformer((pipe_numerical, numerical_features), 
                                                (pipe_categorical,categorical_features), remainder='passthrough')

        return preprocessing
     
    def ridge_model(self, X_train, X_test, y_train, y_test, x1, x2, x3, x4):
        # On définit les variables numériques et catégorielles
        categorical_features = X_train.select_dtypes(include=['object']).columns
        numerical_features = list(X_train.columns.values)
        
        # Mise en place pipeline pour chaque catégorie de variable
        # Pas de NaN pour les valeurs catégorielles donc pas de SimpleImputer
        # Remplacement des NaN valeurs nuémriques par la médiane car nous avons des outliers non gérés
        # Robuscaler() : moins sensible aux valeurs extrêmes

        pipe_categorical = make_pipeline(OneHotEncoder(handle_unknown='ignore'))
        pipe_numerical  = make_pipeline(SimpleImputer(missing_values=np.nan, strategy="median"), RobustScaler())
        # On transforme les colonnes crées avec les pipeline de preprocessing
        # (pipe_categorical,categorical_features)

        preprocessing = make_column_transformer((pipe_numerical, numerical_features), 
                                                (pipe_categorical,categorical_features), remainder='passthrough')
        
        model = make_pipeline(preprocessing, Ridge(alpha=.5, random_state = 32))
        model.fit(X_train, y_train)
        print('---------------------------------------------------------------------------------------------------------------------------')
        print('------------------------------------  Score model sur les Train Sets avec Sklearn  ----------------------------------------')
        print('---------------------------------------------------------------------------------------------------------------------------')
        score_model_train = model.score(X_train, y_train)
        print('\t\t\t\t\t\t R² =', score_model_train)
        print('---------------------------------------------------------------------------------------------------------------------------')
        print("-------------------------------------  Cross Validation : Accuracy - MSE : Train ------------------------------------------")
        print('---------------------------------------------------------------------------------------------------------------------------')
        crossValSCORESTrain = cross_val_score(model, X_train, y_train, cv = 6)
        crossValSCORESTrain = pd.DataFrame(crossValSCORESTrain, columns = ['CV Accuracy Train'])
#         print(crossValSCORESTrain)
        crossValMSETrain = cross_val_score(model,X_train, y_train,
                                           scoring='neg_mean_squared_error', cv = 6)
        crossValMSETrain = pd.DataFrame(crossValMSETrain, columns = ['CV MSE Train'])
#         print(crossValMSETrain)
        cv_train = crossValSCORESTrain.join(crossValMSETrain).T
        print(cv_train)
        print('\n')
        print('---------------------------------------------------------------------------------------------------------------------------')
        print('------------------------------------  Score model sur les Test Sets avec Sklearn  -----------------------------------------')
        print('---------------------------------------------------------------------------------------------------------------------------')
        score_model_test = model.score(X_test, y_test)
        print('\t\t\t\t\t\t R² =', score_model_test)
        print('---------------------------------------------------------------------------------------------------------------------------')
        print("-------------------------------------  Cross Validation : Accuracy - MSE : Test -------------------------------------------")
        print('---------------------------------------------------------------------------------------------------------------------------')
        crossValSCORESTest = cross_val_score(model, X_test, y_test, cv = 6)
        crossValSCORESTest = pd.DataFrame(crossValSCORESTest, columns = ['CV Accuracy Test'])
#         print(crossValSCORESTest)
        crossValMSEtest = cross_val_score(model,X_test, y_test,
                                           scoring='neg_mean_squared_error', cv = 6)
        crossValMSEtest = pd.DataFrame(crossValMSEtest, columns = ['CV MSE Test'])      
#         print(crossValMSEtest)
        cv_test = crossValSCORESTest.join(crossValMSEtest).T
        print(cv_test)
        
        y_pred_train = model.predict(X_train)
        residuals_train = y_pred_train - y_train
        
#         plt.figure(figsize=(15, 5))
#         sns.histplot(data=residuals_train, kde=True)
#         plt.xlim(x1,x2)
#         plt.title('Distribution des résidus de model prediction Trains')
#         plt.show()
# #         print(residuals_train.describe())
#         print('\t\t\t\t\t ', f"Moyenne des résidus : {np.mean(residuals_train)}")
#         print('\t\t\t\t\t ', f"Médiane des résidus : {np.median(residuals_train)}")
#         print('\t\t\t\t\t ', f"Mode des résidus : {mode(residuals_train)}")
        
#         # QQ plot : permet de vérifier si la distribution suit une loi normale
#         plt.figure(figsize=(15, 5))
#         stats.probplot(residuals_train, dist="norm", plot=pylab)
#         pylab.show()

        y_pred_test = model.predict(X_test)
        residuals_test = y_pred_test - y_test
        
        
        
        print('---------------------------------------------------------------------------------------------------------------------------')
        print("-------------------------------------  Optimisation du modèle avec GridSearchCV -------------------------------------------")
        print('---------------------------------------------------------------------------------------------------------------------------')
        # find optimal alpha with grid search
        alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        param_grid = dict(ridge__alpha = alpha)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='r2')
        grid_result = grid.fit(X_train, y_train)
        print('\t\t\t\t\t\t Best Score: ', grid_result.best_score_)
        print('\t\t\t\t\t\t Best Params: ', grid_result.best_params_)
        print('---------------------------------------------------------------------------------------------------------------------------')
        print('------------------------------------  Score modèle GridSearchCV sur les Test Sets  ----------------------------------------')
        print('---------------------------------------------------------------------------------------------------------------------------')
        model_ridge = grid_result.best_estimator_
        score_model = model_ridge.score(X_test, y_test)
        print('\t\t\t\t\t\t R² =', score_model)
        
        print('---------------------------------------------------------------------------------------------------------------------------')
        print("---------------------------------------------  Vérification des hypothèses  ------------------------------------------------")
        print('---------------------------------------------------------------------------------------------------------------------------')
              
        
        plt.figure(figsize=(15, 5))
        sns.histplot(data=residuals_test, kde=True)
        plt.xlim(x3,x4)
        plt.title('Distribution des résidus de model prediction Tests')
        plt.show()
#         print(residuals_test.describe())
        print('\t\t\t\t\t ', f"Moyenne des résidus : {np.mean(residuals_test)}")
        print('\t\t\t\t\t ', f"Médiane des résidus : {np.median(residuals_test)}")
        print('\t\t\t\t\t ', f"Mode des résidus : {mode(residuals_test)}")
        
        # QQ plot : permet de vérifier si la distribution suit une loi normale
        plt.figure(figsize=(15, 5))
        stats.probplot(residuals_test, dist="norm", plot=pylab)
        pylab.show()
        
        # y_pred = model.predict() 
        # residuals = model.resid

        plt.figure(figsize=(15,10))
        sns.regplot(x=y_pred_test, y=residuals_test, lowess=True, line_kws={'color': 'red'})
        plt.title('Mesure de l\'homoscedasticité des résidus', fontsize=16)
        plt.xlabel('Predictions')
        plt.ylabel('Residuals')
        plt.show()
        return cv_train, cv_test, y_pred_train, y_pred_test
    
    
    def lasso_model(self, X_train, X_test, y_train, y_test, x1, x2, x3, x4):
        # On définit les variables numériques et catégorielles
        categorical_features = X_train.select_dtypes(include=['object']).columns
        numerical_features = list(X_train.columns.values)
        
        # Mise en place pipeline pour chaque catégorie de variable
        # Pas de NaN pour les valeurs catégorielles donc pas de SimpleImputer
        # Remplacement des NaN valeurs nuémriques par la médiane car nous avons des outliers non gérés
        # Robuscaler() : moins sensible aux valeurs extrêmes

        pipe_categorical = make_pipeline(OneHotEncoder(handle_unknown='ignore'))
        pipe_numerical  = make_pipeline(SimpleImputer(missing_values=np.nan, strategy="median"), RobustScaler())
        # On transforme les colonnes crées avec les pipeline de preprocessing
        # (pipe_categorical,categorical_features)

        preprocessing = make_column_transformer((pipe_numerical, numerical_features), 
                                                (pipe_categorical,categorical_features), remainder='passthrough')
        
        model = make_pipeline(preprocessing, Lasso(alpha=.5, random_state = 32))
        model.fit(X_train, y_train)
        print('---------------------------------------------------------------------------------------------------------------------------')
        print('------------------------------------  Score model sur les Train Sets avec Sklearn  ----------------------------------------')
        print('---------------------------------------------------------------------------------------------------------------------------')
        score_model_train = model.score(X_train, y_train)
        print('\t\t\t\t\t\t R² =', score_model_train)
        print('---------------------------------------------------------------------------------------------------------------------------')
        print("-------------------------------------  Cross Validation : Accuracy - MSE : Train ------------------------------------------")
        print('---------------------------------------------------------------------------------------------------------------------------')
        crossValSCORESTrain = cross_val_score(model, X_train, y_train, cv = 6)
        crossValSCORESTrain = pd.DataFrame(crossValSCORESTrain, columns = ['CV Accuracy Train'])
#         print(crossValSCORESTrain)
        crossValMSETrain = cross_val_score(model,X_train, y_train,
                                           scoring='neg_mean_squared_error', cv = 6)
        crossValMSETrain = pd.DataFrame(crossValMSETrain, columns = ['CV MSE Train'])
#         print(crossValMSETrain)
        cv_train = crossValSCORESTrain.join(crossValMSETrain).T
        print(cv_train)
        print('\n')
        print('---------------------------------------------------------------------------------------------------------------------------')
        print('------------------------------------  Score model sur les Test Sets avec Sklearn  -----------------------------------------')
        print('---------------------------------------------------------------------------------------------------------------------------')
        score_model_test = model.score(X_test, y_test)
        print('\t\t\t\t\t\t R² =', score_model_test)
        print('---------------------------------------------------------------------------------------------------------------------------')
        print("-------------------------------------  Cross Validation : Accuracy - MSE : Test -------------------------------------------")
        print('---------------------------------------------------------------------------------------------------------------------------')
        crossValSCORESTest = cross_val_score(model, X_test, y_test, cv = 6)
        crossValSCORESTest = pd.DataFrame(crossValSCORESTest, columns = ['CV Accuracy Test'])
#         print(crossValSCORESTest)
        crossValMSEtest = cross_val_score(model,X_test, y_test,
                                           scoring='neg_mean_squared_error', cv = 6)
        crossValMSEtest = pd.DataFrame(crossValMSEtest, columns = ['CV MSE Test'])      
#         print(crossValMSEtest)
        cv_test = crossValSCORESTest.join(crossValMSEtest).T
        print(cv_test)
        
        y_pred_train = model.predict(X_train)
        residuals_train = y_pred_train - y_train
        
#         plt.figure(figsize=(15, 5))
#         sns.histplot(data=residuals_train, kde=True)
#         plt.xlim(x1,x2)
#         plt.title('Distribution des résidus de model prediction Trains')
#         plt.show()
# #         print(residuals_train.describe())
#         print('\t\t\t\t\t ', f"Moyenne des résidus : {np.mean(residuals_train)}")
#         print('\t\t\t\t\t ', f"Médiane des résidus : {np.median(residuals_train)}")
#         print('\t\t\t\t\t ', f"Mode des résidus : {mode(residuals_train)}")
        
#         # QQ plot : permet de vérifier si la distribution suit une loi normale
#         plt.figure(figsize=(15, 5))
#         stats.probplot(residuals_train, dist="norm", plot=pylab)
#         pylab.show()

        y_pred_test = model.predict(X_test)
        residuals_test = y_pred_test - y_test
        
        plt.figure(figsize=(15, 5))
        sns.histplot(data=residuals_test, kde=True)
        plt.xlim(x3,x4)
        plt.title('Distribution des résidus de model prediction Tests')
        plt.show()
#         print(residuals_test.describe())
        print('\t\t\t\t\t ', f"Moyenne des résidus : {np.mean(residuals_test)}")
        print('\t\t\t\t\t ', f"Médiane des résidus : {np.median(residuals_test)}")
        print('\t\t\t\t\t ', f"Mode des résidus : {mode(residuals_test)}")
        
        # QQ plot : permet de vérifier si la distribution suit une loi normale
        plt.figure(figsize=(15, 5))
        stats.probplot(residuals_test, dist="norm", plot=pylab)
        pylab.show()
        print('---------------------------------------------------------------------------------------------------------------------------')
        print("-------------------------------------  Optimisation du modèle avec GridSearchCV -------------------------------------------")
        print('---------------------------------------------------------------------------------------------------------------------------')
        # find optimal alpha with grid search
        alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        param_grid = dict(lasso__alpha = alpha)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='r2')
        grid_result = grid.fit(X_train, y_train)
        print('\t\t\t\t\t\t Best Score: ', grid_result.best_score_)
        print('\t\t\t\t\t\t Best Params: ', grid_result.best_params_)
        print('---------------------------------------------------------------------------------------------------------------------------')
        print('------------------------------------  Score modèle GridSearchCV sur les Test Sets  ----------------------------------------')
        print('---------------------------------------------------------------------------------------------------------------------------')
        model_lasso = grid_result.best_estimator_
        score_model_lasso = model_lasso.score(X_test, y_test)
        print('\t\t\t\t\t\t R² =', score_model_lasso)
        
        return cv_train, cv_test, y_pred_train, y_pred_test

    
    def elasticNetCV_model(self, X_train, X_test, y_train, y_test):
        # On définit les variables numériques et catégorielles
        categorical_features = X_train.select_dtypes(include=['object']).columns
        numerical_features = list(X_train.columns.values)
        
        # Mise en place pipeline pour chaque catégorie de variable
        # Pas de NaN pour les valeurs catégorielles donc pas de SimpleImputer
        # Remplacement des NaN valeurs nuémriques par la médiane car nous avons des outliers non gérés
        # Robuscaler() : moins sensible aux valeurs extrêmes

        pipe_categorical = make_pipeline(OneHotEncoder(handle_unknown='ignore'))
        pipe_numerical  = make_pipeline(SimpleImputer(missing_values=np.nan, strategy="median"), RobustScaler())
        # On transforme les colonnes crées avec les pipeline de preprocessing
        # (pipe_categorical,categorical_features)

        preprocessing = make_column_transformer((pipe_numerical, numerical_features), 
                                                (pipe_categorical,categorical_features), remainder='passthrough')
        
        model = make_pipeline(preprocessing, ElasticNetCV(random_state = 32))
        model.fit(X_train, y_train)
        print('---------------------------------------------------------------------------------------------------------------------------')
        print('------------------------------------  Score model sur les Train Sets avec Sklearn  ----------------------------------------')
        print('---------------------------------------------------------------------------------------------------------------------------')
        score_model_train = model.score(X_train, y_train)
        print('\t\t\t\t\t\t R² =', score_model_train)
        print('---------------------------------------------------------------------------------------------------------------------------')
        print("-------------------------------------  Cross Validation : Accuracy - MSE : Train ------------------------------------------")
        print('---------------------------------------------------------------------------------------------------------------------------')
        crossValSCORESTrain = cross_val_score(model, X_train, y_train, cv = 6)
        crossValSCORESTrain = pd.DataFrame(crossValSCORESTrain, columns = ['CV Accuracy Train'])
#         print(crossValSCORESTrain)
        crossValMSETrain = cross_val_score(model,X_train, y_train,
                                           scoring='neg_mean_squared_error', cv = 6)
        crossValMSETrain = pd.DataFrame(crossValMSETrain, columns = ['CV MSE Train'])
#         print(crossValMSETrain)
        cv_train = crossValSCORESTrain.join(crossValMSETrain).T
        print(cv_train)
        print('\n')
        print('---------------------------------------------------------------------------------------------------------------------------')
        print('------------------------------------  Score model sur les Test Sets avec Sklearn  -----------------------------------------')
        print('---------------------------------------------------------------------------------------------------------------------------')
        score_model_test = model.score(X_test, y_test)
        print('\t\t\t\t\t\t R² =', score_model_test)
        print('---------------------------------------------------------------------------------------------------------------------------')
        print("-------------------------------------  Cross Validation : Accuracy - MSE : Test -------------------------------------------")
        print('---------------------------------------------------------------------------------------------------------------------------')
        crossValSCORESTest = cross_val_score(model, X_test, y_test, cv = 6)
        crossValSCORESTest = pd.DataFrame(crossValSCORESTest, columns = ['CV Accuracy Test'])
#         print(crossValSCORESTest)
        crossValMSEtest = cross_val_score(model,X_test, y_test,
                                           scoring='neg_mean_squared_error', cv = 6)
        crossValMSEtest = pd.DataFrame(crossValMSEtest, columns = ['CV MSE Test'])      
#         print(crossValMSEtest)
        cv_test = crossValSCORESTest.join(crossValMSEtest).T
        print(cv_test)
        
        y_pred_train = model.predict(X_train)
        residuals_train = y_pred_train - y_train
        
#         plt.figure(figsize=(15, 5))
#         sns.histplot(data=residuals_train, kde=True)
# #         plt.xlim(x1,x2)
#         plt.title('Distribution des résidus de model prediction Trains')
#         plt.show()
# #         print(residuals_train.describe())
#         print('\t\t\t\t\t ', f"Moyenne des résidus : {np.mean(residuals_train)}")
#         print('\t\t\t\t\t ', f"Médiane des résidus : {np.median(residuals_train)}")
#         print('\t\t\t\t\t ', f"Mode des résidus : {mode(residuals_train)}")
        
#         # QQ plot : permet de vérifier si la distribution suit une loi normale
#         plt.figure(figsize=(15, 5))
#         stats.probplot(residuals_train, dist="norm", plot=pylab)
#         pylab.show()

        y_pred_test = model.predict(X_test)
        residuals_test = y_pred_test - y_test
        
#         plt.figure(figsize=(15, 5))
#         sns.histplot(data=residuals_test, kde=True)
# #         plt.xlim(x3,x4)
#         plt.title('Distribution des résidus de model prediction Tests')
#         plt.show()
# #         print(residuals_test.describe())
#         print('\t\t\t\t\t ', f"Moyenne des résidus : {np.mean(residuals_test)}")
#         print('\t\t\t\t\t ', f"Médiane des résidus : {np.median(residuals_test)}")
#         print('\t\t\t\t\t ', f"Mode des résidus : {mode(residuals_test)}")
        
#         # QQ plot : permet de vérifier si la distribution suit une loi normale
#         plt.figure(figsize=(15, 5))
#         stats.probplot(residuals_test, dist="norm", plot=pylab)
#         pylab.show()
#         print('---------------------------------------------------------------------------------------------------------------------------')
#         print("-------------------------------------  Optimisation du modèle avec GridSearchCV -------------------------------------------")
#         print('---------------------------------------------------------------------------------------------------------------------------')
#         # Use grid search to tune the parameters:

#         parametersGrid = {'elasticnet__max_iter': [1, 5, 10],
#                           'elasticnet__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
#                           'elasticnet__l1_ratio': np.arange(0.0, 1.0, 0.1)}

# #         eNet = ElasticNet()
#         grid = GridSearchCV(model, parametersGrid, scoring='r2', cv=10)
#         grid_result = grid.fit(X_train, y_train)
#         print('\t\t\t\t\t\t Best Score: ', grid_result.best_score_)
#         print('\t\t\t\t\t\t Best Params: ', grid_result.best_params_)
#         print('---------------------------------------------------------------------------------------------------------------------------')
#         print('------------------------------------  Score modèle GridSearchCV sur les Test Sets  ----------------------------------------')
#         print('---------------------------------------------------------------------------------------------------------------------------')
#         model_eNet = grid.best_estimator_
#         score_model_eNet = model_eNet.score(X_test, y_test)
#         print('\t\t\t\t\t\t R² =', score_model_eNet)
        
        return cv_train, cv_test, y_pred_train, y_pred_test
    