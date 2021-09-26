
metrics = []

with open('artifact.txt', 'r') as f:
    while True:
        line = f.readline()   

        if not line:
            break
        


        if 'TEST METRICS' in line:
            jacc = f.readline()
            if 'jaccard' not in jacc:
                continue
            jacc = jacc.split()[1]
            sens = f.readline().split()[1]
            spec = f.readline().split()[1]
            acc = f.readline().split()[1]
            prec = f.readline().split()[1]

            #print(acc, jacc, prec, sens, spec)
            metrics.append([acc, jacc, prec, sens, spec])

print(len(metrics), metrics)


# ---------------TEST METRICS----------------------
# jaccard_index 0.7802364347819745
# test_sensitivity 0.8481279581687503
# test_specifitivity 0.973312916208177
# test_accuracy 0.9453213739056959
# test_precision 0.9015042411518669
# test_jaccard_score 0.7802364347819745
# test_dicecoef 0.8740019206613738
# isic_eval_score 0.9964539007092199
# ---------------TEST METRICS----------------------