import numpy as np

def cosine_similarity(arr1, arr2):
    ans = []
    arr1, arr2 = arr1.toarray(), arr2.toarray()
    for l1 in arr1:
        a = []
        for l2 in arr2:
            s1,s2 = np.sum(l1), np.sum(l2)
            cosine = 0
            for k in range(l1.shape[0]):
                i, j = l1[k], l2[k]
                cosine += (i*j)
            cosine = cosine/float((s1*s2)**0.5)
            a.append(cosine)
        ans.append(a)
    return np.array(ans)