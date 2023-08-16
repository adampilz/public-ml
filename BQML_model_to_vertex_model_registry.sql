# create a bq dataset
CREATE SCHEMA example_ds
OPTIONS (
    location = 'US'
  , default_table_expiration_days = 1
  , description = 'dataset for this example'
);


CREATE  OR REPLACE TABLE `example_ds.synthetic_data_train` (
  feature1 STRING,
  feature2 FLOAT64,
  mypartition BOOL,
  label FLOAT64
);

INSERT INTO `example_ds.synthetic_data_train`
SELECT
  CASE
    WHEN MOD(CAST(FLOOR(RAND() * 100) AS INT64), 3) = 0 THEN 'A'
    WHEN MOD(CAST(FLOOR(RAND() * 100) AS INT64), 3) = 1 THEN 'B'
    ELSE 'C'
  END AS feature1,
  RAND() AS feature2,
  CASE WHEN RAND() < 0.7 THEN False ELSE True END AS mypartition,
  CASE WHEN RAND() < 0.5 THEN 0 ELSE 1 END AS label
FROM
  UNNEST(GENERATE_ARRAY(1, 10000))
;


# train xgb binary classification model
CREATE OR REPLACE MODEL `example_ds.xgboost_model`
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER'
  , input_label_cols=['label']
  , data_split_method = "CUSTOM"
  , data_split_col = "mypartition"
  , MODEL_REGISTRY = "VERTEX_AI"
  , VERTEX_AI_MODEL_ID = "xgboost_model"
  , VERTEX_AI_MODEL_VERSION_ALIASES = ["v2"]
  ) AS
SELECT
  feature1,
  feature2,
  mypartition,
  label
FROM
  `example_ds.synthetic_data_train`
;