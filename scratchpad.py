#%%

results = {
    'chart.a': [
        {'dim.a': {'score': 0.167}},
        {'sent': {'score': 0.15}},
        {'delivered': {'score': 0.1678}}
    ],
    'chart.b': [
        {'dim.a': {'score': 0.14394685039}},
        {'dim.b': {'score': 0.22}}
    ]
}
print(results)

#%%

# get max and min scores
results_list = []
for chart in results:
    for dimension in results[chart]:
        for k in dimension:
            results_list.append([chart, k, dimension[k]['score']])

print(results_list)

#%%