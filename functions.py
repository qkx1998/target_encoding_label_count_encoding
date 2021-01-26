def n_fold_target_encoding(train_df,test_df,label='label',n=5,enc_list=[],functions=['mean']):
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=2)
    for f in tqdm(enc_list):
        for func in functions:
            train_df[f + f'_target_enc_{func}'] = 0
            test_df[f + f'_target_enc_{func}'] = 0
            for i, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df[label])):
                trn_x = train_df[[f, label]].iloc[trn_idx].reset_index(drop=True)
                val_x = train_df[[f]].iloc[val_idx].reset_index(drop=True)
                enc_df = trn_x.groupby(f, as_index=False)[label].agg({f + f'_target_enc_{func}': func})
                val_x = val_x.merge(enc_df, on=f, how='left')
                test_x = test_df[[f]].merge(enc_df, on=f, how='left')
                val_x[f + f'_target_enc_{func}'] = val_x[f + f'_target_enc_{func}'].fillna(train_df[label].agg(func))
                test_x[f + f'_target_enc_{func}'] = test_x[f + f'_target_enc_{func}'].fillna(train_df[label].agg(func))
                train_df.loc[val_idx, f + f'_target_enc_{func}'] = val_x[f + f'_target_enc_{func}'].values
                test_df[f + f'_target_enc_{func}'] += test_x[f + f'_target_enc_{func}'].values / skf.n_splits
    return train_df,test_df

def labelcount_encode(X, categorical_features, ascending=False):
    print('LabelCount encoding: {}'.format(categorical_features))
    X_ = pd.DataFrame()
    for cat_feature in categorical_features:
        cat_feature_value_counts = X[cat_feature].value_counts()
        value_counts_list = cat_feature_value_counts.index.tolist()
        if ascending:
            value_counts_range = list(
                reversed(range(len(cat_feature_value_counts))))
        else:
            value_counts_range = list(range(len(cat_feature_value_counts)))
        labelcount_dict = dict(zip(value_counts_list, value_counts_range))
        X_[cat_feature] = X[cat_feature].map(
            labelcount_dict)
    X_ = X_.add_suffix('_labelcount_encoded')
    if ascending:
        X_ = X_.add_suffix('_ascending')
    else:
        X_ = X_.add_suffix('_descending')
    X_ = X_.astype(np.uint32)
    return X_
