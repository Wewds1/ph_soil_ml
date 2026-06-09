import pandas as pd
import numpy as np
import os

class SoilDataPipeline:
    """
    Professional pipeline for soil data cleaning and preprocessing.
    Implements BSWM standards for Philippine soil data.
    """

    @staticmethod
    def professional_clean_season(s):
        s = str(s).strip().lower()
        if any(w in s for w in ['wet', 'rainy']): return 'Wet Season'
        if 'dry' in s: return 'Dry Season'
        return 'Transition'

    @staticmethod
    def clean_drainage(s):
        drain_map = {
            'poorly drained': 'Poorly Drained',
            'somewhat poorly drained': 'Somewhat Poorly Drained',
            'moderately well drained': 'Moderately Well Drained',
            'well drained': 'Well Drained',
            'excessively drained': 'Excessively Drained',
        }
        s = str(s).strip().lower()
        return drain_map.get(s, 'Moderately Well Drained')

    def preprocess(self, df):
        """Full preprocessing pipeline from raw data to cleaned dataframe."""
        df = df.copy()

        # 1. Missingness flags
        missing_cols = ['phosphorus_ppm', 'organic_matter_pct', 'cation_exchange_capacity', 'base_saturation_pct', 'soil_depth_cm']
        for col in missing_cols:
            df[f'{col}_missing'] = df[col].isnull().astype(int)

        # 2. Domain-specific imputation
        # Phosphorus: group by texture
        df['phosphorus_ppm'] = df['phosphorus_ppm'].fillna(df.groupby('soil_texture_class')['phosphorus_ppm'].transform('median'))

        # Organic Matter: N * 10 rule
        df['organic_matter_pct'] = df['organic_matter_pct'].fillna(df['nitrogen_pct'] * 10).clip(0.5, 8.0)

        # CEC: group by texture
        df['cation_exchange_capacity'] = df['cation_exchange_capacity'].fillna(df.groupby('soil_texture_class')['cation_exchange_capacity'].transform('median'))

        # Base Saturation: group by pH buckets
        df['ph_bucket'] = pd.cut(df['soil_ph'], bins=[0, 5.5, 7.0, 14], labels=['Acidic', 'Neutral', 'Alkaline'])
        df['base_saturation_pct'] = df['base_saturation_pct'].fillna(df.groupby('ph_bucket')['base_saturation_pct'].transform('median'))
        df = df.drop(columns=['ph_bucket'])

        # Soil Depth: mode (75)
        df['soil_depth_cm'] = df['soil_depth_cm'].fillna(75)

        # 3. Categorical cleaning
        df['season'] = df['season'].apply(self.professional_clean_season)
        df['drainage_class'] = df['drainage_class'].apply(self.clean_drainage)

        # General text standardization
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols:
            if col not in ['season', 'drainage_class']:
                df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True).str.strip().str.title()

        # 4. Deduplication
        df = df.drop_duplicates(subset=[c for c in df.columns if c != 'sample_id'])

        return df

class SoilFeatureEngineer:
    """
    Advanced feature engineering for Philippine soil suitability.
    Translates raw chemistry into agronomic indicators.
    """

    def transform(self, df):
        df = df.copy()

        # --- Block 2: NPK Interpretation ---
        # Nitrogen rating
        df['n_rating'] = pd.cut(df['nitrogen_pct'],
                               bins=[0, 0.10, 0.20, 1.0],
                               labels=['Low', 'Medium', 'High'])

        # Phosphorus rating (Olsen method)
        df['p_rating'] = pd.cut(df['phosphorus_ppm'],
                               bins=[0, 10, 25, 200],
                               labels=['Low', 'Medium', 'High'])

        # Potassium rating
        df['k_rating'] = pd.cut(df['potassium_cmol'],
                               bins=[0, 0.2, 0.5, 10],
                               labels=['Low', 'Medium', 'High'])

        # pH Interpretation
        df['ph_class'] = pd.cut(df['soil_ph'],
                               bins=[0, 4.5, 5.5, 6.5, 7.5, 14],
                               labels=['Strongly Acidic', 'Moderately Acidic', 'Slightly Acidic', 'Near Neutral', 'Alkaline'])

        # Overall Fertility Index (Composite)
        df['fertility_index'] = (
            (df['nitrogen_pct'] / 0.20).clip(0, 1) * 0.30 +
            (df['phosphorus_ppm'] / 25).clip(0, 1) * 0.25 +
            (df['potassium_cmol'] / 0.5).clip(0, 1) * 0.20 +
            (df['organic_matter_pct'] / 3.0).clip(0, 1) * 0.25
        ).round(3)

        # --- Block 3: Soil Health and Structural Features ---
        # pH deviation from ideal (6.2)
        df['ph_deviation_from_ideal'] = (df['soil_ph'] - 6.2).abs().round(2)

        # Compaction flag
        df['is_compacted'] = (df['bulk_density_gcc'] > 1.4).astype(int)

        # Rooting depth adequacy
        df['shallow_soil'] = (df['soil_depth_cm'] < 45).astype(int)
        df['deep_soil']    = (df['soil_depth_cm'] > 90).astype(int)

        # Drainage ordinal
        drain_ord = {'Poorly Drained': 1, 'Somewhat Poorly Drained': 2,
                     'Moderately Well Drained': 3, 'Well Drained': 4, 'Excessively Drained': 5}
        df['drainage_ordinal'] = df['drainage_class'].map(drain_ord).fillna(3)

        # Organic matter adequacy
        df['om_adequate'] = (df['organic_matter_pct'] >= 2.0).astype(int)

        # --- Block 4: Climate and Seasonal Interactions ---
        # Agroclimatic zone: climate_type + elevation bin
        elev_bins = pd.cut(df['elevation_m'], bins=[-1, 200, 600, 1000, 3000], labels=['Lowland', 'Midland', 'Highland', 'Upland'])
        df['agroclimatic_zone'] = df['climate_type'].astype(str) + '_' + elev_bins.astype(str)

        # Peak growing period (Type I + Wet Season)
        df['peak_growing_period'] = ((df['climate_type'] == 'Type I') & (df['season'] == 'Wet Season')).astype(int)

        # Highland zone (Elev > 800m + Temp < 24)
        df['highland_zone'] = ((df['elevation_m'] > 800) & (df['temp_c'] < 24)).astype(int)

        # Irrigated potential
        df['irrigated_potential'] = (
            (df['slope_pct'] < 5) &
            (df['near_water_body'] == 1) &
            df['drainage_class'].isin(['Moderately Well Drained', 'Well Drained'])
        ).astype(int)

        # Temperature suitability
        df['lowland_heat'] = (df['temp_c'] > 27).astype(int)
        df['highland_cool'] = (df['temp_c'] < 22).astype(int)

        # --- Block 5: Farm History Features ---
        # Soil depletion risk
        df['soil_depletion_risk'] = (df['farm_history'] == 'Monocrop').astype(int)

        # Prior legume (N boost)
        df['prior_legume'] = df['previous_crop'].isin(['Sitaw', 'Munggo', 'Vegetable']).astype(int)

        # Virgin land bonus
        df['virgin_land_bonus'] = (df['farm_history'] == 'Virgin Land').astype(int)

        # Farm history ordinal score
        fh_ord = {'Virgin Land': 5, 'Rotation': 4, 'Fallow': 3, 'Orchard': 2, 'Monocrop': 1}
        df['farm_history_score'] = df['farm_history'].map(fh_ord).fillna(3)

        # --- Block 6: Log transforms for skewed variables ---
        df['log_phosphorus'] = np.log1p(df['phosphorus_ppm'])
        df['log_rainfall'] = np.log1p(df['rainfall_mm'])
        df['log_elevation'] = np.log1p(df['elevation_m'])
        df['log_slope'] = np.log1p(df['slope_pct'])

        return df
