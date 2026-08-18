"""
02_preprocessing.py

Netflix Quantum ML Mini Project
--------------------------------
Step 1. Netflix Dataset Load
Step 2. Target Encoding
Step 3. release_year Feature 생성
Step 4. rating Encoding
Step 5. genre_count 생성
Step 6. country_count 생성
Step 7. description_length 생성
Step 8. Feature Candidate 확인
Step 9. Quantum-ready Feature 선택
Step 10. Data Quality 확인
Step 11. Preprocessed Dataset 저장

Target
------
Movie   -> 0
TV Show -> 1

Feature Candidates
------------------
release_year_feature
rating_encoded
genre_count
country_count
description_length

기본 Quantum-ready Features
---------------------------
release_year_feature
rating_encoded
genre_count
description_length

주의
----
duration은 Movie / TV Show 정보를 지나치게 직접적으로
포함할 수 있으므로 기본 Feature에서 제외한다.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = Path("netflix_titles.csv")
OUTPUT_PATH = Path("data/netflix_preprocessed.csv")


# Quantum ML에서 기본적으로 사용할 Feature
QUANTUM_FEATURES = [
    "release_year_feature",
    "rating_encoded",
    "genre_count",
    "description_length",
]


# 전체 Feature Engineering 후보
FEATURE_CANDIDATES = [
    "release_year_feature",
    "rating_encoded",
    "genre_count",
    "country_count",
    "description_length",
]


# ============================================================
# Utility
# ============================================================

def print_section(title: str) -> None:
    """실습 출력 결과를 단계별로 구분한다."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# STEP 1. Dataset Load
# ============================================================

def load_dataset(data_path: Path) -> pd.DataFrame:
    """Netflix 원본 Dataset을 불러온다."""

    print_section("STEP 1. Netflix Dataset Load")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset을 찾을 수 없습니다: {data_path.resolve()}"
        )

    df = pd.read_csv(data_path)

    print(f"Dataset Path : {data_path}")
    print(f"Rows         : {df.shape[0]:,}")
    print(f"Columns      : {df.shape[1]}")

    return df


# ============================================================
# STEP 2. Target Encoding
# ============================================================

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classification Target인 type을 숫자로 변환한다.

    Movie   -> 0
    TV Show -> 1
    """

    print_section("STEP 2. Target Encoding")

    if "type" not in df.columns:
        raise KeyError("'type' column이 존재하지 않습니다.")

    df = df.copy()

    target_mapping = {
        "Movie": 0,
        "TV Show": 1,
    }

    df["target"] = df["type"].map(target_mapping)

    print("Target Mapping")
    print("--------------")
    print("Movie   -> 0")
    print("TV Show -> 1")

    print()
    print("[Target Distribution]")

    print(
        df[["type", "target"]]
        .value_counts()
        .sort_index()
    )

    unmapped = df["target"].isnull().sum()

    if unmapped > 0:
        print()
        print(
            f"Warning: Target으로 변환되지 않은 데이터가 "
            f"{unmapped:,}건 있습니다."
        )

    return df


# ============================================================
# STEP 3. release_year Feature
# ============================================================

def create_release_year_feature(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    release_year를 Numeric Feature로 사용한다.
    """

    print_section("STEP 3. release_year Feature")

    if "release_year" not in df.columns:
        raise KeyError(
            "'release_year' column이 존재하지 않습니다."
        )

    df = df.copy()

    df["release_year_feature"] = pd.to_numeric(
        df["release_year"],
        errors="coerce",
    )

    print(
        df[
            [
                "release_year",
                "release_year_feature",
            ]
        ].head(10)
    )

    return df


# ============================================================
# STEP 4. Rating Encoding
# ============================================================

def create_rating_feature(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    rating 문자열을 Category Code로 변환한다.

    교육용 Mini Project이므로 간단한 정수 Encoding을 사용한다.

    주의:
    단순 정수 Encoding은 Category 사이에 실제로 존재하지 않는
    순서 관계를 암묵적으로 만들 수 있다.
    """

    print_section("STEP 4. Rating Encoding")

    if "rating" not in df.columns:
        raise KeyError(
            "'rating' column이 존재하지 않습니다."
        )

    df = df.copy()

    print("[Original Rating Distribution]")

    print(
        df["rating"]
        .value_counts(dropna=False)
    )

    # 실행 순서와 무관하게 항상 같은 코드가 나오도록
    # 정렬된 category 목록 사용
    rating_values = sorted(
        df["rating"]
        .dropna()
        .astype(str)
        .unique()
    )

    rating_mapping = {
        rating: index
        for index, rating in enumerate(rating_values)
    }

    df["rating_encoded"] = (
        df["rating"]
        .map(rating_mapping)
        .fillna(-1)
        .astype(int)
    )

    print()
    print("[Rating Mapping]")

    for rating, code in rating_mapping.items():
        print(
            f"{rating:<15} -> {code}"
        )

    print()
    print("Missing Rating -> -1")

    return df, rating_mapping


# ============================================================
# STEP 5. genre_count
# ============================================================

def create_genre_count(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    listed_in에 포함된 장르 개수를 계산한다.

    Example
    -------
    "Dramas, International Movies"
            ->
             2
    """

    print_section("STEP 5. Genre Count")

    if "listed_in" not in df.columns:
        raise KeyError(
            "'listed_in' column이 존재하지 않습니다."
        )

    df = df.copy()

    df["genre_count"] = (
        df["listed_in"]
        .fillna("")
        .apply(
            lambda value: len(
                [
                    genre.strip()
                    for genre in value.split(",")
                    if genre.strip()
                ]
            )
        )
    )

    print(
        df[
            [
                "listed_in",
                "genre_count",
            ]
        ].head(10)
    )

    return df


# ============================================================
# STEP 6. country_count
# ============================================================

def create_country_count(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    country에 포함된 제작 국가의 개수를 계산한다.

    Example
    -------
    United States
        -> 1

    United States, Canada
        -> 2
    """

    print_section("STEP 6. Country Count")

    if "country" not in df.columns:
        raise KeyError(
            "'country' column이 존재하지 않습니다."
        )

    df = df.copy()

    df["country_count"] = (
        df["country"]
        .fillna("")
        .apply(
            lambda value: len(
                [
                    country.strip()
                    for country in value.split(",")
                    if country.strip()
                ]
            )
        )
    )

    print(
        df[
            [
                "country",
                "country_count",
            ]
        ].head(10)
    )

    return df


# ============================================================
# STEP 7. description_length
# ============================================================

def create_description_length(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    description의 단어 개수를 Numeric Feature로 변환한다.

    전체 Text를 Quantum Circuit에 입력하는 대신
    간단한 Numeric Feature로 압축한다.
    """

    print_section("STEP 7. Description Length")

    if "description" not in df.columns:
        raise KeyError(
            "'description' column이 존재하지 않습니다."
        )

    df = df.copy()

    df["description_length"] = (
        df["description"]
        .fillna("")
        .astype(str)
        .apply(
            lambda value: len(value.split())
        )
    )

    print(
        df[
            [
                "description",
                "description_length",
            ]
        ].head(10)
    )

    return df


# ============================================================
# STEP 8. Feature Candidate Review
# ============================================================

def review_features(
    df: pd.DataFrame,
) -> None:
    """생성된 Feature 후보를 확인한다."""

    print_section("STEP 8. Feature Candidate Review")

    print("Feature Candidates")
    print("------------------")

    for index, feature in enumerate(
        FEATURE_CANDIDATES,
        start=1,
    ):
        print(
            f"{index}. {feature}"
        )

    print()
    print("[Feature Preview]")

    print(
        df[
            FEATURE_CANDIDATES
            + ["target"]
        ].head(10)
    )

    print()
    print("[Feature Data Types]")

    print(
        df[
            FEATURE_CANDIDATES
            + ["target"]
        ].dtypes
    )


# ============================================================
# STEP 9. Quantum-ready Feature Selection
# ============================================================

def select_quantum_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Quantum ML 실험에 사용할 기본 Feature를 선택한다.

    4 Features
        ->
    4-dimensional Input

    실제 Qubit 수는 Feature Map의 Encoding 방식에 따라 결정된다.
    """

    print_section(
        "STEP 9. Quantum-ready Feature Selection"
    )

    print("Selected Features")
    print("-----------------")

    for index, feature in enumerate(
        QUANTUM_FEATURES,
        start=1,
    ):
        print(
            f"{index}. {feature}"
        )

    X = df[QUANTUM_FEATURES].copy()
    y = df["target"].copy()

    print()
    print(f"X Shape : {X.shape}")
    print(f"y Shape : {y.shape}")

    print()
    print("[X Preview]")

    print(X.head())

    print()
    print("[y Preview]")

    print(y.head())

    return X, y


# ============================================================
# STEP 10. Data Quality Check
# ============================================================

def check_preprocessed_data(
    df: pd.DataFrame,
) -> None:
    """전처리 결과의 결측치와 데이터 타입을 확인한다."""

    print_section(
        "STEP 10. Preprocessed Data Quality Check"
    )

    check_columns = (
        FEATURE_CANDIDATES
        + ["target"]
    )

    print("[Missing Values]")

    print(
        df[check_columns]
        .isnull()
        .sum()
    )

    print()
    print("[Data Types]")

    print(
        df[check_columns]
        .dtypes
    )

    print()
    print("[Basic Statistics]")

    print(
        df[check_columns]
        .describe()
        .round(2)
    )


# ============================================================
# STEP 11. Create Output Dataset
# ============================================================

def create_output_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """최종 Preprocessed Dataset을 생성한다."""

    print_section(
        "STEP 11. Create Preprocessed Dataset"
    )

    output_columns = [
        "show_id",
        "title",
        "type",
        "target",
        "release_year_feature",
        "rating_encoded",
        "genre_count",
        "country_count",
        "description_length",
    ]

    # 혹시 target/release_year 등에 결측치가 있다면
    # 모델 입력 전에 제거한다.
    preprocessed_df = (
        df[output_columns]
        .copy()
    )

    before_rows = len(preprocessed_df)

    required_columns = [
        "target",
        "release_year_feature",
        "rating_encoded",
        "genre_count",
        "country_count",
        "description_length",
    ]

    preprocessed_df = (
        preprocessed_df
        .dropna(subset=required_columns)
        .reset_index(drop=True)
    )

    # target은 Binary Integer로 확정
    preprocessed_df["target"] = (
        preprocessed_df["target"]
        .astype(int)
    )

    after_rows = len(preprocessed_df)

    print(
        f"Before Cleaning : {before_rows:,}"
    )

    print(
        f"After Cleaning  : {after_rows:,}"
    )

    print(
        f"Removed Rows    : "
        f"{before_rows - after_rows:,}"
    )

    print()
    print("[Preprocessed Dataset]")

    print(
        preprocessed_df.head(10)
    )

    return preprocessed_df


# ============================================================
# STEP 12. Save Dataset
# ============================================================

def save_dataset(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """전처리 Dataset을 CSV 파일로 저장한다."""

    print_section(
        "STEP 12. Save Preprocessed Dataset"
    )

    # data 디렉터리가 없으면 생성
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved : {output_path}"
    )

    print(
        f"Rows  : {len(df):,}"
    )

    print(
        f"Columns : {df.shape[1]}"
    )


# ============================================================
# STEP 13. Final Summary
# ============================================================

def show_summary(
    df: pd.DataFrame,
) -> None:
    """전처리 결과를 최종 요약한다."""

    print_section(
        "STEP 13. Preprocessing Summary"
    )

    print(
        f"Dataset Size : "
        f"{df.shape[0]:,} rows"
    )

    print(
        f"Columns      : "
        f"{df.shape[1]}"
    )

    print()
    print("Target")
    print("------")
    print("Movie   -> 0")
    print("TV Show -> 1")

    print()
    print("Quantum-ready Features")
    print("----------------------")

    for feature in QUANTUM_FEATURES:
        print(
            f"- {feature}"
        )

    print()
    print("Excluded Feature")
    print("----------------")

    print(
        "duration -> Data Leakage 가능성 때문에 "
        "기본 Classification Feature에서 제외"
    )

    print()
    print("Next Step")
    print("---------")

    print(
        "03_quantum_sample.py에서 Quantum 실험을 위한 "
        "Sampling과 데이터 구성을 수행합니다."
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print_section(
        "Netflix Quantum ML Mini Project - Preprocessing"
    )

    # STEP 1
    df = load_dataset(INPUT_PATH)

    # STEP 2
    df = encode_target(df)

    # STEP 3
    df = create_release_year_feature(df)

    # STEP 4
    df, rating_mapping = create_rating_feature(df)

    # STEP 5
    df = create_genre_count(df)

    # STEP 6
    df = create_country_count(df)

    # STEP 7
    df = create_description_length(df)

    # STEP 8
    review_features(df)

    # STEP 9
    X, y = select_quantum_features(df)

    # STEP 10
    check_preprocessed_data(df)

    # STEP 11
    preprocessed_df = create_output_dataset(df)

    # STEP 12
    save_dataset(
        preprocessed_df,
        OUTPUT_PATH,
    )

    # STEP 13
    show_summary(
        preprocessed_df,
    )


if __name__ == "__main__":
    main()