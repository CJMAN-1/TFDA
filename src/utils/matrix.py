
def gram_mat(x):
    (b, c, h, w) = x.size()
    f = x.view(b, c, h*w)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (c*w*h)
    return G