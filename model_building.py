# import relevant libraries

# for data visualization 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# for data cleanzing and manipulation 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# for resampling technique
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline

# for model building
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

# for model evaluation 
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report, confusion_matrix

# for hyperparameters tuning
from sklearn.model_selection import GridSearchCV

###############################################
########## 1. Read the CSV file ###############
###############################################
# define a filepath 
filepath = r"C:\Users\jeffr\Desktop\mental_health_disorder\dataset\dataset.csv"

# use pandas function to read the csv file
mental_disorder_df = pd.read_csv(filepath)

# read the first ten rows of the data
print(mental_disorder_df.head(10))


#################################################
########## 2. Data Preprocessing  ###############
#################################################

# 2.1 Data Information 
print(mental_disorder_df.info())

# visualize the missing data using heatmap
sns.heatmap(mental_disorder_df.isnull(), annot = False, cmap = 'viridis')

# count the number of missing data in each column (i.e., attribute) 
print(mental_disorder_df.isnull().sum())

# rename columns 
mental_disorder_df.rename(columns = {'feeling.nervous': 'feeling_nervous',
                                     'breathing.rapidly': 'breathing_rapidly',
                                     'trouble.in.concentration': 'trouble_concentration',
                                     'having.trouble.in.sleeping': 'trouble_sleeping',
                                     'having.trouble.with.work': 'trouble_with_work',
                                     'over.react': 'overreact',
                                     'change.in.eating': 'change_in_eating',
                                     'suicidal.thought': 'suicidal_thought',
                                     'feeling.tired': 'feeling_tired',
                                     'close.friend': 'close_friend',
                                     'social.media.addiction': 'social_media_addiction',
                                     'weight.gain': 'weight_gain',
                                     'material.possessions': 'material_possessions',
                                     'popping.up.stressful.memory': 'popping_up_stressful_memory',
                                     'having.nightmares': 'having_nightmares',
                                     'avoids.people.or.activities': 'avoid_people_or_activities',
                                     'feeling.negative': 'feeling_negative',
                                     'trouble.concentrating': 'trouble_concentrating',
                                     'blamming.yourself': 'blaming_yourself',
                                     'Disorder': 'disorder'}, inplace = True)

# review rename columns
print(mental_disorder_df.info())

# remove the trouble concentrating columns
mental_disorder_df.drop(columns = ['trouble_concentrating'], axis = 1, inplace = True)

# get the unique list of disorder
disorder_list = mental_disorder_df['disorder'].unique()

########################################################
########## 3. Exploratory Data Analysis  ###############
########################################################

# create a countplot for respective attributes with respect to the mental disorders
plt.figure(figsize = (12, 5))
sns.set_style("darkgrid")
sns.countplot(x = mental_disorder_df['feeling_nervous'], hue = mental_disorder_df['disorder'])
plt.title("Number of People Feeling Nervous with Respect to Mental Disorder", fontweight = 'bold', fontsize = 15)
plt.xlabel("Feeling Nervous?", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of Observations", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor=(1.02, 0.65))
plt.show()

plt.figure(figsize = (12, 5))
sns.set_style("darkgrid")
sns.countplot(x = mental_disorder_df['breathing_rapidly'], hue = mental_disorder_df['disorder'])
plt.title("Number of People Having Breathing Difficulties with Respect to Mental Disorder", fontweight = 'bold', fontsize = 15)
plt.xlabel("Feeling Breathing Difficulties?", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of Observations", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor=(1.02, 0.65))
plt.show()

plt.figure(figsize = (12, 5))
sns.set_style("darkgrid")
sns.countplot(x = mental_disorder_df['trouble_sleeping'], hue = mental_disorder_df['disorder'])
plt.title("Number of People Trouble Sleeping with Respect to Mental Disorder", fontweight = 'bold', fontsize = 15)
plt.xlabel("Trouble Sleeping?", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of Observations", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor=(1.02, 0.65))
plt.show()

plt.figure(figsize = (12, 5))
sns.set_style("darkgrid")
sns.countplot(x = mental_disorder_df['trouble_concentration'], hue = mental_disorder_df['disorder'])
plt.title("Number of People Trouble Concentrations with Respect to Mental Disorder", fontweight = 'bold', fontsize = 15)
plt.xlabel("Trouble Concentrations?", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of Observations", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor=(1.02, 0.65))
plt.show()


plt.figure(figsize = (12, 5))
sns.set_style("darkgrid")
sns.countplot(x = mental_disorder_df['trouble_with_work'], hue = mental_disorder_df['disorder'])
plt.title("Number of People Trouble with Work with Respect to Mental Disorder", fontweight = 'bold', fontsize = 15)
plt.xlabel("Trouble with Work?", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of Observations", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor=(1.02, 0.65))
plt.show()

plt.figure(figsize = (12, 5))
sns.set_style("darkgrid")
sns.countplot(x = mental_disorder_df['overreact'], hue = mental_disorder_df['disorder'])
plt.title("Number of People Over-React with Respect to Mental Disorder", fontweight = 'bold', fontsize = 15)
plt.xlabel("OverReact?", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of Observations", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor=(1.02, 0.65))
plt.show()

plt.figure(figsize = (12, 5))
sns.set_style("darkgrid")
sns.countplot(x = mental_disorder_df['overreact'], hue = mental_disorder_df['disorder'])
plt.title("Number of People with Changing Eating Habits with Respect to Mental Disorder", fontweight = 'bold', fontsize = 15)
plt.xlabel("Changing Eating Habits?", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of Observations", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor=(1.02, 0.65))
plt.show()

plt.figure(figsize = (12, 5))
sns.set_style("darkgrid")
sns.countplot(x = mental_disorder_df['suicidal_thought'], hue = mental_disorder_df['disorder'])
plt.title("Number of People with Suicidal Thought Habits with Respect to Mental Disorder", fontweight = 'bold', fontsize = 15)
plt.xlabel("With Suicidal Thought?", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of Observations", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor=(1.02, 0.65))
plt.show()

plt.figure(figsize = (12, 5))
sns.set_style("darkgrid")
sns.countplot(x = mental_disorder_df['feeling_tired'], hue = mental_disorder_df['disorder'])
plt.title("Number of People Feeling Tired Habits with Respect to Mental Disorder", fontweight = 'bold', fontsize = 15)
plt.xlabel("Feeling Tired?", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of Observations", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor=(1.02, 0.65))
plt.show()

plt.figure(figsize = (12, 5))
sns.set_style("darkgrid")
sns.countplot(x = mental_disorder_df['close_friend'], hue = mental_disorder_df['disorder'])
plt.title("Number of People with Close Friends with Respect to Mental Disorder", fontweight = 'bold', fontsize = 15)
plt.xlabel("Have Close Friends?", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of Observations", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor=(1.02, 0.65))
plt.show()

plt.figure(figsize = (12, 5))
sns.set_style("darkgrid")
sns.countplot(x = mental_disorder_df['social_media_addiction'], hue = mental_disorder_df['disorder'])
plt.title("Number of People with Social Media Addiction with Respect to Mental Disorder", fontweight = 'bold', fontsize = 15)
plt.xlabel("Social Media Addiction?", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of Observations", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor=(1.02, 0.65))
plt.show()

plt.figure(figsize = (12, 5))
sns.set_style("darkgrid")
sns.countplot(x = mental_disorder_df['weight_gain'], hue = mental_disorder_df['disorder'])
plt.title("Number of People with Gaining of Weight Addiction with Respect to Mental Disorder", fontweight = 'bold', fontsize = 15)
plt.xlabel("Weight Gain?", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of Observations", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor=(1.02, 0.65))
plt.show()

plt.figure(figsize = (12, 5))
sns.set_style("darkgrid")
sns.countplot(x = mental_disorder_df['material_possessions'], hue = mental_disorder_df['disorder'])
plt.title("Number of People with Material Possessions with Respect to Mental Disorder", fontweight = 'bold', fontsize = 15)
plt.xlabel("Material Possessions?", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of Observations", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor=(1.02, 0.65))
plt.show()

plt.figure(figsize = (12, 5))
sns.set_style("darkgrid")
sns.countplot(x = mental_disorder_df['popping_up_stressful_memory'], hue = mental_disorder_df['disorder'])
plt.title("Number of People with Popping Up Stressful Memory with Respect to Mental Disorder", fontweight = 'bold', fontsize = 15)
plt.xlabel("Popping Up Stressful Memory?", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of Observations", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor=(1.02, 0.65))
plt.show()

plt.figure(figsize = (12, 5))
sns.set_style("darkgrid")
sns.countplot(x = mental_disorder_df['having_nightmares'], hue = mental_disorder_df['disorder'])
plt.title("Number of People with Nightmares with Respect to Mental Disorder", fontweight = 'bold', fontsize = 15)
plt.xlabel("Having Nightmares?", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of Observations", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor=(1.02, 0.65))
plt.show()

plt.figure(figsize = (12, 5))
sns.set_style("darkgrid")
sns.countplot(x = mental_disorder_df['avoid_people_or_activities'], hue = mental_disorder_df['disorder'])
plt.title("Number of People who avoid people or other social activities with Respect to Mental Disorder", fontweight = 'bold', fontsize = 15)
plt.xlabel("Avoid people or activities?", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of Observations", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor=(1.02, 0.65))
plt.show()

plt.figure(figsize = (12, 5))
sns.set_style("darkgrid")
sns.countplot(x = mental_disorder_df['feeling_negative'], hue = mental_disorder_df['disorder'])
plt.title("Number of People Feeling Negative with Respect to Mental Disorder", fontweight = 'bold', fontsize = 15)
plt.xlabel("Feeling Negative?", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of Observations", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor=(1.02, 0.65))
plt.show()

plt.figure(figsize = (12, 5))
sns.set_style("darkgrid")
sns.countplot(x = mental_disorder_df['blaming_yourself'], hue = mental_disorder_df['disorder'])
plt.title("Number of People Blaming Themselves with Respect to Mental Disorder", fontweight = 'bold', fontsize = 15)
plt.xlabel("Blaming Yourself?", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of Observations", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor=(1.02, 0.65))
plt.show()


##############################################################
########## 4. Encode the Categorical Features  ###############
##############################################################
# obtain the column name
col_names = mental_disorder_df.columns
print(col_names)

# define label encoder object
le = LabelEncoder()

# replace the categorical value into numerical value for cat features
for cat in col_names:
    mental_disorder_df[cat] = le.fit_transform(mental_disorder_df[cat])


#############################################
########## 5. Data Splitting  ###############
#############################################
X_train, X_test, y_train, y_test = train_test_split(mental_disorder_df.drop(columns = ['disorder'], axis = 1),
                                                    mental_disorder_df['disorder'],
                                                    test_size = 0.2, 
                                                    random_state = 42, shuffle = True)


#####################################################################
############### 6. Spot Check on ML Algorithms  #####################
#####################################################################
def get_ML_models():
    models_, names_ = list(), list()
    
    # random forest classifier
    rfc = RandomForestClassifier(random_state = 42)
    models_.append(rfc)
    names_.append('random forest')
    
    # gradient boosting classifier
    gbc = GradientBoostingClassifier(random_state = 42)
    models_.append(gbc)
    names_.append('gradient boosting')
    
    # extra trees classifier
    etc = ExtraTreesClassifier(random_state = 42)
    models_.append(etc)
    names_.append('extra tress')
    
    return models_, names_

def evaluate_models(model, X, y):
    # define cv parameters 
    cv = ShuffleSplit(train_size = 0.8, test_size = 0.2, n_splits = 5, random_state = 42)
    # define roc-auc-scoring 
    roc_auc_scoring = make_scorer(roc_auc_score, multi_class = 'ovr', needs_proba = True)
    # get the CV score
    score = cross_validate(model, X, y, cv = cv, scoring = roc_auc_scoring, n_jobs = 1, return_train_score = True)
    return score    

# call the function to get the ML models
models, names = get_ML_models()

for i in range(len(models)):
    # build and fit the model
    model = models[i]
    model.fit(X_train, y_train)
    
    # evaluate the training model
    scoring = evaluate_models(model, X_train, y_train)
    print(">> {}: {:.3f}({:.3f})".format(names[i], scoring['train_score'].mean(), scoring['train_score'].std()))


#############################################################################
############### 7. Spot Check on Resampling Techniques  #####################
#############################################################################
def get_resampling_models():
    models_, names_ = list(), list()
    
    # SMOTE-Tomek
    smote_tomek = SMOTETomek(random_state = 42)
    models_.append(smote_tomek)
    names_.append('smote-tomek')
    
    # SMOTE-ENN
    smote_enn = SMOTEENN(random_state = 42)
    models_.append(smote_enn)
    names_.append('smote-enn')
    
    # SMOTE
    smote = SMOTE(random_state = 42)
    models_.append(smote)
    names_.append('smote')
    
    # Borderline SMOTE
    smote_borderline = BorderlineSMOTE(random_state = 42)
    models_.append(smote_borderline)
    names_.append('smote-borderline')
    
    # Tomek links
    tomek_links = TomekLinks()
    models_.append(tomek_links)
    names_.append('tomek_links')
    
    return models_, names_


def evaluate_models(model, X, y):
    # define cv parameters 
    cv = ShuffleSplit(train_size = 0.8, test_size = 0.2, n_splits = 5, random_state = 42)
    # define roc-auc-scoring 
    roc_auc_scoring = make_scorer(roc_auc_score, multi_class = 'ovr', needs_proba = True)
    # get the CV score
    score = cross_validate(model, X, y, cv = cv, scoring = roc_auc_scoring, n_jobs = 1, return_train_score = True)
    return score    


# define the ML models
gbc = GradientBoostingClassifier(random_state = 42)

# call the function to get the resampling models
models, names = get_resampling_models()

for i in range(len(models)):
    # define pipeline steps
    steps = [('resampling', models[i]), ('model', gbc)]
    # define pipeline
    pipeline = Pipeline(steps = steps)
    
    # evaluate the training model
    scoring = evaluate_models(pipeline, X_train, y_train)
    print(">> {}: {:.3f}({:.3f})".format(names[i], scoring['train_score'].mean(), scoring['train_score'].std()))


#######################################################################
############### 8. Resampling the Training Data   #####################
#######################################################################
# distribution of imbalanced classification on training set
plt.figure(figsize = (12, 5))
sns.set_style("darkgrid")
sns.countplot(x = y_train)
plt.title("Distribution of Imbalanced Classification of Mental Disorder (Training Set)", fontsize = 15, fontweight = 'bold')
plt.xlabel("Disorder", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of Observations", fontweight = 'bold', fontsize = 12)
plt.show()


# balanced the target classification using TomekLinks
smoteenn = SMOTEENN(random_state = 42)
x_smt, y_smt = smoteenn.fit_resample(X_train, y_train)


# distribution of balanced classification on training set
plt.figure(figsize = (12, 5))
sns.set_style("darkgrid")
sns.countplot(x = y_smt)
plt.title("Distribution of Balanced Classification of Mental Disorder (Training Set)", fontsize = 15, fontweight = 'bold')
plt.xlabel("Disorder", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of Observations", fontweight = 'bold', fontsize = 12)
plt.show()





###############################################################################################
############### 9. Build GBC Model using balanced training set   ##############################
###############################################################################################
# define gbc model parameters
gbc_model = GradientBoostingClassifier(n_estimators = 500,
                                       learning_rate = 0.01,
                                       max_depth = 3,
                                       random_state = 42)

# fit the model using resampled training data
gbc_model.fit(x_smt, y_smt)

############################################################################
############### 10. Evaluate the  GBC Model    ############################
############################################################################

# predict the y-train and y_test data
predict_y_train = gbc_model.predict(x_smt)
predict_y_test = gbc_model.predict(X_test)

# classification report for training set
print("Classification Report for Training Set:")
print(classification_report(y_smt, predict_y_train))

# classification report for testing set
print("Classification Report for Testing Set:")
print(classification_report(y_test, predict_y_test))
print()

# confusion matrix for training set
plt.subplot(1, 2, 1)
cnf_matrix_training = confusion_matrix(y_smt, predict_y_train)
sns.heatmap(cnf_matrix_training, annot = True, cmap = 'icefire')
plt.title("Confusion Matrix for Training Set", fontweight = 'bold', fontsize = 12)

plt.subplot(1, 2, 2)
cnf_matrix_testing = confusion_matrix(y_test, predict_y_test)
sns.heatmap(cnf_matrix_testing, annot = True, cmap = 'icefire')
plt.title("Confusion Matrix for Testing Set", fontweight = 'bold', fontsize = 12)
plt.tight_layout()
plt.show()
print()

# evaluation metrics for training set
print("Evaluation Metrics for Training Set: ")
print("Accuracy: {:.2f}".format(accuracy_score(y_smt, predict_y_train)))
print("Precision: {:.2f}".format(precision_score(y_smt, predict_y_train, average = 'micro')))
print("F1-Score: {:.2f}".format(f1_score(y_smt, predict_y_train, average = 'micro')))
print("Recall: {:.2f}".format(recall_score(y_smt, predict_y_train, average = 'micro')))

# evaluation metrics for testing set
print("Evaluation Metrics for Testing Set: ")
print("Accuracy: {:.2f}".format(accuracy_score(y_test, predict_y_test)))
print("Precision: {:.2f}".format(precision_score(y_test, predict_y_test, average = 'macro')))
print("F1-Score: {:.2f}".format(f1_score(y_test, predict_y_test, average = 'macro')))
print("Recall: {:.2f}".format(recall_score(y_test, predict_y_test, average = 'macro')))


################################################################################
############### 11. Hyperparameters Tuning on GBC Model    #####################
################################################################################
# define gbc parameters to be tuned
# params = {'n_estimators': [10, 100, 150, 250, 500],
#           'max_depth': [1, 3, 5, 7, 9], 
#           'learning_rate': [0.01, 0.1, 1, 10, 100]}

# # define model parameters
# gbc_model_grid = GradientBoostingClassifier(random_state = 42)

# # define cross-validation parameters
# cv = ShuffleSplit(train_size = 0.8, test_size = 0.2, n_splits = 5, random_state = 42)

# # define the grid search parameters
# grid_cv = GridSearchCV(gbc_model_grid, param_grid = params, cv = cv, scoring = 'accuracy', verbose = 3, return_train_score = True, n_jobs = 1)
# grid_cv.fit(x_smt, y_smt)

# print("Best Params: {}".format(grid_cv.best_params_))
# print("Best Score: {}".format(grid_cv.best_score_))


##########################################################
############### 12. Save the Model   #####################
##########################################################
import pickle

# define filepath 
filepath_ = r"C:\Users\jeffr\Desktop\mental_health_disorder\gbc_model.sav"
pickle.dump(gbc_model, open(filepath_, "wb"))














