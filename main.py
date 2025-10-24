from itertools import combinations
import pandas as pd
import numpy as np
import os

# Import generalized toolkit and configuration files
import ml_toolkit as mlt
import style_config as sC # Style config is used indirectly via toolkit
import pet_config as pC   # Data-specific configurations

def load_and_validate_data(data_path, cols_of_interest):
    """Load the raw data and select columns of interest."""
    print(f"Loading data from: {data_path}")
    try:
        data_frame_in = pd.read_csv(data_path)
        
        if 'Diagnosis' not in data_frame_in.columns:
            raise ValueError("'Diagnosis' column missing in raw data.")
            
        data_frame_in = data_frame_in[cols_of_interest].dropna(subset=['Diagnosis'])
        
        print(f"Data loaded with shape: {data_frame_in.shape}")
        return data_frame_in
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}. Please update the path in pet_config.py.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return pd.DataFrame()

def run_analysis(analysis_key, analysis_config, global_config, data_frame_raw):
    """Executes a single analysis configuration (overview or pairwise)."""
    
    config_plots = global_config['plotting_parameters']
    config_switches = global_config['analysis_switches']
    
    # 1. Apply row-wise mapping (e.g., combining tumor subtypes into groups)
    data_frame_grouped = mlt.tailor_data_frame_rows(
        column_of_interest="Diagnosis", 
        data_frame_in=data_frame_raw, 
        mapping_dict=analysis_config['grouping_map'],
    )
    
    # 2. Dispatch based on analysis type
    if analysis_config['analysis_type'] == 'overview':
        print(f"  Running Overview Analysis: {analysis_config['name']}")
        result_path = analysis_config['result_path']
        os.makedirs(result_path, exist_ok=True)
        
        # General Violin/Box plots
        if config_switches['plot']:
            mlt.plot_violin_diagnosis(
                category_column="Diagnosis",
                data_frame_in=data_frame_grouped,
                covariate=analysis_config.get('covariate', None),
                result_path=result_path, 
                show_dots=True,
                fig_size=sC.fig_size_5,
            )
        
        # Dimensionality Reduction Plots (using ALL features)
        # Assuming 'Diagnosis' is the only non-feature column at this stage.
        num_label_cols = len(global_config['columns_of_interest']) - (data_frame_raw.shape[1] - len(data_frame_raw.columns.difference(global_config['columns_of_interest'])))
        
        mlt.plot_dimensionality_reduction( 
            outcome_columns_to_plot=["Diagnosis"],
            data_frame_in=data_frame_grouped,
            impute_data=config_plots['impute_data'],
            max_num_features=data_frame_grouped.shape[1] - 1, # Use all features
            num_label_columns=1, 
            plot_pca=config_plots['plot_pca'],
            plot_tsne=config_plots['plot_tsne'],
            result_path=result_path,
            plot_strategy_name=analysis_config['name'],
        )

    elif analysis_config['analysis_type'] == 'pairwise':
        groups = analysis_config['tumour_groups']
        combinations_to_run = pC.get_pairwise_combinations(groups)
        
        print(f"  Running Pairwise Analysis on {len(groups)} groups ({len(combinations_to_run)} combinations)...")
        
        for group_a, group_b in combinations_to_run:
            pair_name = f"{group_a}-{group_b}"
            print(f"    Processing combination: {pair_name}")
            result_path = os.path.join(analysis_config['result_path_base'], pair_name)
            os.makedirs(result_path, exist_ok=True)
            
            # 2.1 Filter data for the current pair (removes other groups and NaN feature columns)
            data_frame_pair = mlt.tailor_data_frame_cols(
                column_of_interest="Diagnosis", 
                comparison_groups=[group_a, group_b],
                data_frame_in=data_frame_grouped, 
            )
            
            if data_frame_pair.empty or data_frame_pair.shape[0] < 3:
                print(f"      Skipped: Insufficient data for {pair_name}.")
                continue
            
            num_label_cols = 1
            
            # 2.2 Uni-variate AUC Analysis and Feature Ranking
            data_frame_sorted, data_frame_auc, data_frame_na_p, best_feature_idx = mlt.calculate_auc(
                data_frame_in=data_frame_pair, 
                outcome_columns=["Diagnosis"],
                cross_validation=config_plots['cross_validation_method'],
                classifier_type=config_plots['classifier_type'],
                result_path=result_path,
                num_label_columns=num_label_cols,
                rank_by_column="Diagnosis",
                overwrite_switch=config_plots['overwrite_switch'],
            )
            
            # 2.3 Multi-variate AUC Analysis (Feature selection trend)
            idx_max_auc_dict = mlt.calculate_multi_auc(
                auc_display_min=config_plots['auc_display_min'],
                data_frame_in=data_frame_sorted, 
                cross_validation=config_plots['cross_validation_method'],
                num_label_columns=num_label_cols,
                max_num_features=config_plots['max_num_features'],
                result_path=result_path,
                overwrite_switch=config_plots['overwrite_switch'],
                classifier_type=config_plots['classifier_type'],
            )
            
            idx_max_auc = idx_max_auc_dict.get("Diagnosis", 0) 
            
            # 2.4 Visualization
            if config_switches['plot']:
                # Violin plots
                mlt.plot_violin_diagnosis(
                    data_frame_in=data_frame_pair,
                    result_path=result_path,
                    show_dots=True,
                    fig_size=sC.fig_size_2,
                )
                
                # Dimensionality Reduction Plots (using top N features)
                mlt.plot_dimensionality_reduction( 
                    outcome_columns_to_plot=["Diagnosis"],
                    data_frame_in=data_frame_sorted, 
                    auc_data_frame=data_frame_auc,
                    na_percent_data_frame=data_frame_na_p,
                    impute_data=config_plots['impute_data'],
                    max_num_features=config_plots['max_num_features'],
                    num_label_columns=num_label_cols,
                    plot_pca=config_plots['plot_pca'],
                    plot_tsne=config_plots['plot_tsne'],
                    result_path=result_path,
                    plot_strategy_name=pair_name,
                )

            # 2.5 Multi-class Machine Learning (LOOCV)
            if config_switches['loocv']:
                # Use features up to the index corresponding to the max MAUC index
                data_frame_loocv = data_frame_sorted.iloc[:, :num_label_cols + idx_max_auc + 1] 
                
                mlt.execute_loocv(
                    category_column="Diagnosis", 
                    data_frame_in=data_frame_loocv, 
                    result_path=result_path,
                    n_repeat=config_plots['n_repeat_loocv'],
                    title=pair_name + f" (Top {data_frame_loocv.shape[1] - num_label_cols} Features)",
                )


def main():
    """Main execution function for the generalized analysis pipeline."""
    
    global_config = pC.GLOBAL_CONFIG
    
    # 1. Initial Data Load
    data_frame_raw = load_and_validate_data(
        global_config['data_path'], 
        global_config['columns_of_interest']
    )
    if data_frame_raw.empty:
        return
    
    # 2. Run enabled analyses defined in pet_config.py
    for sw_name, analysis_config in pC.ANALYSIS_CONFIGS.items():
        if global_config['analysis_switches'].get(sw_name, False):
            print(f"\n==============================================")
            print(f"RUNNING ANALYSIS: {analysis_config['name']}")
            print(f"==============================================")
            run_analysis(sw_name, analysis_config, global_config, data_frame_raw)

    print("\n\n=== Generalized Classification and Visualization Pipeline Complete! ===")

if __name__ == "__main__":
    main()
