import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from constants import TRAIN_DATA_PATH, TEST_DATA_PATH
from settings import TARGET_COLUMN, NEGATIVE_CLASS, POSITIVE_CLASS
from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_maps = {}

    def fit(self, X, y=None):
        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            self.freq_maps[col] = freq
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in X.columns:
            X_transformed[col] = X_transformed[col].map(self.freq_maps[col]).fillna(0)
        return X_transformed

def load_and_prepare_data():
    columns = ["age", "class_of_worker", "industry_code", "occupation_code", "education", "wage_per_hour",
                "enrolled_in_edu_inst_last_wk", "marital_status", "major_industry_code", "major_occupation_code",
                "race", "hispanic_origin", "sex", "member_of_labor_union", "reason_for_unemployment",
                "full_or_part_time_employment_stat", "capital_gains", "capital_losses", "dividends_from_stocks",
                "tax_filer_status", "region_of_previous_residence", "state_of_previous_residence",
                "detailed_household_and_family_stat", "detailed_household_summary_in_household",
                "instance_weight", "migration_code_change_in_msa", "migration_code_change_in_reg",
                "migration_code_move_within_reg", "live_in_this_house_1_year_ago", "migration_prev_res_in_sunbelt",
                "num_persons_worked_for_employer", "family_members_under_18", "country_of_birth_father",
                "country_of_birth_mother", "country_of_birth_self", "citizenship", "own_business_or_self_employed",
                "fill_inc_questionnaire_for_veterans_admin", "veterans_benefits", "weeks_worked_in_year", "year", "income"
                ]

    train_df = pd.read_csv(TRAIN_DATA_PATH, header=None)
    test_df = pd.read_csv(TEST_DATA_PATH, header=None)
    train_df.columns = test_df.columns = columns

    # Drop instance_weight
    train_df.drop(columns=['instance_weight'], inplace=True)
    test_df.drop(columns=['instance_weight'], inplace=True)

    # Remove feature-only duplicates
    feature_cols = [col for col in train_df.columns if col != TARGET_COLUMN]
    train_df = train_df[~train_df.duplicated(subset=feature_cols)]
    test_df = test_df[~test_df.duplicated(subset=feature_cols)]

    # Map target values
    train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].replace({NEGATIVE_CLASS: 0, POSITIVE_CLASS: 1})
    test_df[TARGET_COLUMN] = test_df[TARGET_COLUMN].replace({NEGATIVE_CLASS: 0, POSITIVE_CLASS: 1})

    # Define features
    num_features = ['age', 'wage_per_hour', 'capital_gains', 'capital_losses',
                    'dividends_from_stocks', 'num_persons_worked_for_employer',
                    'weeks_worked_in_year'
                    ]
    cat_features = [col for col in train_df.columns if col not in num_features + [TARGET_COLUMN]]

    # Explicitly prioritize features identified via EDA
    important_low_card = ["education", "class_of_worker", "major_occupation_code"]
    important_high_card = ["occupation_code"]

    # Dynamically determine cardinality of remaining features
    cat_cardinality = train_df[cat_features].nunique()
    auto_low_card = cat_cardinality[cat_cardinality <= 15].index.difference(important_low_card + important_high_card).tolist()
    auto_high_card = cat_cardinality[cat_cardinality > 15].index.difference(important_low_card + important_high_card).tolist()

    low_card_cat_features = important_low_card + auto_low_card
    high_card_cat_features = important_high_card + auto_high_card

    # Encoding + Scaling
    numeric_transformer = StandardScaler()
    low_card_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    high_card_transformer = FrequencyEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('low_cat', low_card_transformer, low_card_cat_features),
            ('high_cat', high_card_transformer, high_card_cat_features)
        ]
    )

    return train_df, test_df, preprocessor
