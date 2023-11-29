


# create an imbalanced synthetic training dataset suitable for binary classification
CREATE OR REPLACE TABLE `YOUR-PROJECT.YOUR-BQ-DS.synthetic_data_train` AS
SELECT
  ROW_NUMBER() OVER() AS row_id
  , RAND() AS feature1
  , RAND() AS feature2
  , IF(RAND() < 0.8, 0, 1) AS label
FROM
  UNNEST(GENERATE_ARRAY(1, 1000)) AS id
;

# train xgb binary classification model
CREATE OR REPLACE MODEL `YOUR-PROJECT.YOUR-BQ-DS.xgboost_model`
OPTIONS(model_type='BOOSTED_TREE_CLASSIFIER', input_label_cols=['label']) AS
SELECT
  feature1,
  feature2,
  label
FROM
  `YOUR-PROJECT.YOUR-BQ-DS.synthetic_data_train`
;


# create an imbalanced synthetic validation dataset to score using the xgb model
CREATE OR REPLACE TABLE `YOUR-PROJECT.YOUR-BQ-DS.synthetic_data_validation` AS
SELECT
  ROW_NUMBER() OVER() AS row_id
  , RAND() AS feature1
  , RAND() AS feature2
  , IF(RAND() < 0.8, 0, 1) AS label
FROM
  UNNEST(GENERATE_ARRAY(1, 1000)) AS id
;

# score synthetic validation dataset using the xgb model
CREATE OR REPLACE TABLE `YOUR-PROJECT.YOUR-BQ-DS.synthetic_validation_scored` AS
SELECT
    feature1
  , feature2
  , label
  , predicted_label_probs[SAFE_OFFSET(0)].prob AS pos_class_pred_prob
  , predicted_label
FROM
  ML.PREDICT(MODEL `YOUR-PROJECT.YOUR-BQ-DS.xgboost_model`,
    (SELECT *
     FROM `YOUR-PROJECT.YOUR-BQ-DS.synthetic_data_validation`))
;




##############################################################################
#
# train a temperature model - only need one of the next two queries
#
##############################################################################
# without HP tuning
CREATE OR REPLACE MODEL
  `YOUR-PROJECT.YOUR-BQ-DS.temperature_model`
OPTIONS
  ( MODEL_TYPE='LOGISTIC_REG'
    , AUTO_CLASS_WEIGHTS=TRUE
    , MAX_ITERATIONS=2
    , INPUT_LABEL_COLS=["predicted_label"] # change if needed
    , CALCULATE_P_VALUES = TRUE
    , CATEGORY_ENCODING_METHOD="DUMMY_ENCODING"
    , L1_REG=0
     ) AS
SELECT
  # Convert predicted probabilities to logits
  LOG(pos_class_pred_prob / (1 - pos_class_pred_prob)) AS logits
  , predicted_label
FROM `YOUR-PROJECT.YOUR-BQ-DS.synthetic_validation_scored`
;


# with HP tuning
CREATE OR REPLACE MODEL
  `YOUR-PROJECT.YOUR-BQ-DS.temperature_model`
OPTIONS
  ( MODEL_TYPE='LOGISTIC_REG'
    , INPUT_LABEL_COLS = ["predicted_label"] # change if needed
    , AUTO_CLASS_WEIGHTS = TRUE
    , CALCULATE_P_VALUES = TRUE
    , CATEGORY_ENCODING_METHOD = "DUMMY_ENCODING"
    
    # HP tuning
    , NUM_TRIALS = 10 # 10*num_HPs
    , EARLY_STOP = TRUE
    , MAX_ITERATIONS = 10
    , MAX_PARALLEL_TRIALS = 1
    , HPARAM_TUNING_ALGORITHM = "VIZIER_DEFAULT"
    , HPARAM_TUNING_OBJECTIVES = ["LOG_LOSS"] # matches AG
    , L1_REG = HPARAM_CANDIDATES([0]) # matches AG
    , L2_REG = HPARAM_RANGE(0, 10)
     ) AS
SELECT
  # Convert predicted probabilities to logits
  LOG(pos_class_pred_prob / (1 - pos_class_pred_prob)) AS logits
  , predicted_label
FROM `YOUR-PROJECT.YOUR-BQ-DS.synthetic_validation_scored`
;


##############################################################################
#
# view the temperaure coefficient
#
##############################################################################

# If you used HP tuning, click on the model in your BQ dataset, then look at the Evaluation tab.
# this will tell you the optimal trial_id to put in the WHERE statement below.
# if you did not use HP tuning, run the below query without the WHERE statement.

# In both cases, you want the value from the "weight" column in the row that is "logits" from the "processed_input" column

SELECT
  *
FROM
  ML.ADVANCED_WEIGHTS(MODEL `YOUR-PROJECT.YOUR-BQ-DS.temperature_model`
  , STRUCT(FALSE AS standardize)
  )
  
WHERE trial_id = 8 # the optimal value of "weight" column in my case -> 4.091302364453



##############################################################################
#
# create an imblanaced synthetic test dataset to score using the xgb model
#
##############################################################################

CREATE OR REPLACE TABLE `YOUR-BQ-DS.synthetic_data_test` AS
SELECT
  ROW_NUMBER() OVER() AS row_id
  , RAND() AS feature1
  , RAND() AS feature2
  , IF(RAND() < 0.8, 0, 1) AS label
FROM
  UNNEST(GENERATE_ARRAY(1, 100)) AS id
;
;

##############################################################################
#
# apply prediction and temperature scaling
#
##############################################################################

# use xgb model to predict probabilities for test set.
CREATE OR REPLACE TABLE `YOUR-BQ-DS.synthetic_test_scored` AS
SELECT
  predicted_label_probs[SAFE_OFFSET(0)].prob AS pos_class_pred_prob
FROM
  ML.PREDICT(MODEL `YOUR-BQ-DS.xgboost_model`,
    (SELECT *
     FROM `YOUR-BQ-DS.synthetic_data_test`))
;


# apply temperature scaling
SELECT
  *
  , 1 / (1 + EXP(-(LOG(pos_class_pred_prob / (1 - pos_class_pred_prob)) / T))) AS calibrated_probability
FROM `YOUR-BQ-DS.synthetic_test_scored`
  , (SELECT 4.091302364453 AS T)  # this is the value from the last step