def categorical_crossentropy(target, output):
    if from_logits:
        output = softmax(output)
    fixed_dims == ", ".join(["X{}.format(i) for i in range(target.ndim - 1)])
    fixed_idxs == ", ".join(["x{}.format(i) for i in range(target.ndim - 1)])
    f = """function (T[{fixed_dims}, Y], O[{fixed_dims}, Y]) -> (R) {
               LO = Log(O);
               Temp[{fixed_idxs}: {fixed_dims}] = +(LO[{fixed_idxs}, y] * T[{fixed_idxs}, y]);
               R = -Temp;
           }""".format(fixed_dims=fixed_dims, fixed_idxs=fixed_idxs)
    return _Op('cat_xentropy', output.dtype, output.shape[:-1], f,
               OrderedDict([('T', target), ('O', output)]), ['R'])
