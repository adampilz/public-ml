

# create a bq dataset
CREATE SCHEMA example_ds
OPTIONS (
    location = 'US'
  , default_table_expiration_days = 1
  , description = 'dataset for this example'
);


# create a synthetic training dataset suitable for binary classification
CREATE OR REPLACE TABLE `example_ds.synthetic_data_train` AS
SELECT
  ROW_NUMBER() OVER() AS row_id
  , RAND() AS feature1
  , RAND() AS feature2
  , IF(RAND() < 0.5, "product_1", "product_2") AS product_type
  , IF(RAND() < 0.7 , False, True) AS mypartition
  , IF(RAND() < 0.5, 0, 1) AS label
FROM
  UNNEST(GENERATE_ARRAY(1, 10000)) AS id
;


# train xgb binary classification model
CREATE OR REPLACE MODEL `example_ds.xgboost_model`
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER'
  , input_label_cols=['label']
  , data_split_method = "CUSTOM"
  , data_split_col = "mypartition"
  ) AS
SELECT
  feature1,
  feature2,
  label
FROM
  `example_ds.synthetic_data_train`
;


# create a synthetic training dataset suitable for binary classification
CREATE OR REPLACE TABLE `example_ds.synthetic_data_test` AS
SELECT
    RAND() AS feature1
  , RAND() AS feature2
  , IF(RAND() < 0.5, "product_1", "product_2") AS product_type
  , IF(RAND() < 0.5, 0, 1) AS label
FROM
  UNNEST(GENERATE_ARRAY(1, 1000)) AS id
;


# score synthetic test data using the xgb model
CREATE OR REPLACE TABLE `example_ds.synthetic_test_scored` AS
SELECT
    feature1
  , feature2
  , product_type
  , label
  , predicted_label_probs[SAFE_OFFSET(0)].prob AS pos_class_pred_prob
  , predicted_label
FROM
  ML.PREDICT(MODEL `example_ds.xgboost_model`,
    (SELECT *
     FROM `example_ds.synthetic_data_test`
     )
  )
;


###########################################################
#
#  calculate optimal cutoff using cost matrix
#
###########################################################

WITH cutoffs AS (
    -- Generate a series of potential cutoffs between 0 and 1, incremented by 0.01
    SELECT ROUND(x * 0.01, 2) AS cutoff
    FROM UNNEST(GENERATE_ARRAY(0, 100)) AS x
),

confusion_matrix AS (
    -- Compute the TN, FP, FN, TP values for each cutoff and product type
    SELECT
        cutoff,
        product_type,
        COUNTIF(label = 0 AND pos_class_pred_prob <= cutoff) AS TN,
        COUNTIF(label = 0 AND pos_class_pred_prob > cutoff) AS FP,
        COUNTIF(label = 1 AND pos_class_pred_prob <= cutoff) AS FN,
        COUNTIF(label = 1 AND pos_class_pred_prob > cutoff) AS TP
    FROM
        `example_ds.synthetic_test_scored`, cutoffs
    GROUP BY
        cutoff, product_type
),

metrics AS (
    -- Compute accuracy metrics and total cost for each cutoff and product type
    SELECT
        cutoff,
        product_type,
        TN, FP, FN, TP,
        -- Accuracy (This won't produce a division by zero error as the denominator sums counts which can't be all zero together)
        (TP + TN) / (TP + TN + FP + FN) AS accuracy,
        -- Precision
        CASE
            WHEN TP + FP = 0 THEN NULL
            ELSE TP / (TP + FP)
        END AS precision,
        -- Recall/Sensitivity
        CASE
            WHEN TP + FN = 0 THEN NULL
            ELSE TP / (TP + FN)
        END AS recall,
        -- Specificity
        CASE
            WHEN TN + FP = 0 THEN NULL
            ELSE TN / (TN + FP)
        END AS specificity,
        -- F1 Score
        CASE
            WHEN TP + FP = 0 OR TP + FN = 0 THEN NULL
            ELSE 2 * (TP / (TP + FP) * TP / (TP + FN)) / (TP / (TP + FP) + TP / (TP + FN))
        END AS f1_score,
        -- Total Cost using the cost matrix, changing based on product_type
        CASE
            WHEN product_type = 'product_1' THEN (1 * TN) + (2 * FP) + (3 * FN) + (4 * TP)
            WHEN product_type = 'product_2' THEN (2 * TN) + (3 * FP) + (4 * FN) + (5 * TP)
            -- Add more product types and their respective cost matrix as needed
            ELSE (0 * TN) + (1 * FP) + (5 * FN) + (0 * TP)
        END AS total_cost
    FROM
        confusion_matrix
)

-- Find the metrics for the cutoff with the minimum cost for each product type
SELECT
    product_type,
    cutoff,
    accuracy,
    precision,
    recall,
    specificity,
    f1_score,
    total_cost
FROM
    metrics
QUALIFY ROW_NUMBER() OVER (PARTITION BY product_type ORDER BY total_cost ASC) = 1;