import os, warnings

input_path = "/scratch/engin_root/engin/yugtmath/Median_Satellite_View/1/"

for image_name in os.listdir(input_path):
    lst=image_name.split('_')
    if len(lst)==3:
        pass
    elif len(lst)==4:
        assert(lst[1]==' satellite')
        del lst[1]
    else:
        raise AssertionError
    new_name = '{}_satellite_{}_{}'.format(lst[0], lst[1], lst[2])
    print("rename %30s to %30s" % (image_name, new_name))
    os.rename(input_path+image_name, input_path+new_name)