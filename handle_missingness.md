 1. Drop the imputer entirely, pass NaN directly. XGBoost's tree_method='hist' and CatBoost both have native NaN routing. This is the single           
    highest-leverage change.                                                                                                        
    2. Per-feature-group missingness indicators (already present as _missing_rate columns) — these are what matter, not the row sum.                          
    3. had_fibroscan / had_fibrotest binary flags — missingness in NIT biomarkers is clinically meaningful (patients who never got FibroScan are different
    from those who did). These are already derivable from the data and partially captured.                                                            
    4. Pattern-level indicators — e.g., missing_nit_only (NIT missing but biochemistry complete) vs missing_labs — captures whether a patient was in a        
    monitoring protocol vs had incomplete workup. More clinical signal than a count. 