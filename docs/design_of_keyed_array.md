## Design of Keyed Array


### Comparison

| Data structure           | Access by keys<br/>(labels)                                                                                                              | Access by indexes<br/>(integer positions) | Slicing           | 
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|-------------------|
| nested dictionary (dict) | `dict['k1']['k2'][k3]`<br/>`dict.get('k1').get('k2').get('k3')`                                                                          | X                                         | X                 | 
| pandas dataframe (df)    | `df.loc[('k1','k2'),'k3'`                                                                                                                | `df.iloc[row, col]`                       | `df.iloc[1:3, :]` | 
| numpy ndarray (arr)      | X                                                                                                                                        | `arr[i1,i2,i3]`                           | `arr[1:3, :]`     |
| keyed array (karr)       | `karr.loc['k1','k2','k3']`<br/>`karr.get('k1','k2','k3',default)`<br/>`karr.get(dim1_name='k1',dim1_name3='k3',dim1_name2='k2',default)` | `karr[i1,i2,i3]`                          | `karr[1:3, :]`    |


### API
* `values`
* `ndim`
* `dim_names`
* `size`
* `shape`
* `dtype`
* `loc[keys]`
* `key_pos_pairs`
* `key_to_pos(dim: int | str, key: Any, if_not_found = None)`
* `get(*args, default=None, **kwargs)`
