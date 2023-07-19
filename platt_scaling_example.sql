
# EXAMPLE: calibrating predicting probabilities from a tree based model

# create a synthetic training dataset suitable for binary classification
CREATE OR REPLACE TABLE `ds_uscentral1.synthetic_data_train` AS
SELECT
  ROW_NUMBER() OVER() AS row_id
  , RAND() AS feature1
  , RAND() AS feature2
  , IF(RAND() < 0.1, 0, 1) AS label
FROM
  UNNEST(GENERATE_ARRAY(1, 1000)) AS id
;

# train xgb binary classification model
CREATE OR REPLACE MODEL `ds_uscentral1.xgboost_model`
OPTIONS(model_type='BOOSTED_TREE_CLASSIFIER', input_label_cols=['label']) AS
SELECT
  feature1,
  feature2,
  label
FROM
  `ds_uscentral1.synthetic_data_train`
;


# create a synthetic validation dataset to score using the xgb model
CREATE OR REPLACE TABLE `ds_uscentral1.synthetic_data_validation` AS
SELECT
  ROW_NUMBER() OVER() AS row_id
  , RAND() AS feature1
  , RAND() AS feature2
  , IF(RAND() < 0.1, 0, 1) AS label
FROM
  UNNEST(GENERATE_ARRAY(1, 1000)) AS id
;

# score synthetic validation dataset using the xgb model
CREATE OR REPLACE TABLE `ds_uscentral1.synthetic_validation_scored` AS
SELECT
    feature1
  , feature2
  , label
  , predicted_label_probs[SAFE_OFFSET(0)].prob AS pos_class_pred_prob
  , predicted_label
FROM
  ML.PREDICT(MODEL `ds_uscentral1.xgboost_model`,
    (SELECT *
     FROM `ds_uscentral1.synthetic_data_validation`))
;


# train a logistic regression model for platt scaling - probability calibration model
# Use the pos class predicted probs as inputs and the actual labels from the validation dataset as the target
CREATE OR REPLACE MODEL `ds_uscentral1.probability_calibration_model`
OPTIONS(model_type='LOGISTIC_REG', auto_class_weights=TRUE, input_label_cols=['label']) AS
SELECT
  pos_class_pred_prob
  , label
FROM
  `ds_uscentral1.synthetic_validation_scored`
;


# create a synthetic test dataset to score using the xgb model
CREATE OR REPLACE TABLE `ds_uscentral1.synthetic_data_test` AS
SELECT
  ROW_NUMBER() OVER() AS row_id
  , RAND() AS feature1
  , RAND() AS feature2
  , IF(RAND() < 0.1, 0, 1) AS label
FROM
  UNNEST(GENERATE_ARRAY(1, 100)) AS id
;

########################################
# apply platt scaling to the test set
########################################

# use xgb model to predict probabilities for test set.
CREATE OR REPLACE TABLE `ds_uscentral1.synthetic_test_scored` AS
SELECT
  predicted_label_probs[SAFE_OFFSET(0)].prob AS pos_class_pred_prob
FROM
  ML.PREDICT(MODEL `ds_uscentral1.xgboost_model`,
    (SELECT *
     FROM `ds_uscentral1.synthetic_data_test`))
;

# apply the calibration model to adjust these probabilities.
CREATE OR REPLACE TABLE `ds_uscentral1.calibrated_probabilities` AS
SELECT
  pos_class_pred_prob
  , predicted_label_probs[SAFE_OFFSET(0)].prob AS calibrated_pos_class_pred_prob
FROM
  ML.PREDICT(MODEL `ds_uscentral1.probability_calibration_model`,
    (SELECT *
     FROM `ds_uscentral1.synthetic_test_scored`))
;

SELECT
  ROUND(pos_class_pred_prob, 2) AS pos_class_pred_prob
  , ROUND(calibrated_pos_class_pred_prob, 2) AS calibrated_pos_class_pred_prob
FROM `ds_uscentral1.calibrated_probabilities`
;





