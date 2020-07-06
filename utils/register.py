
# Utility function to set parameters of functions.
# Useful to define what consist of a model, dataset, etc.
def wrap_setattr(attr, value):
    def foo(func):
        setattr(func, attr, value)
        return func
    return foo

def setmodelname(value):
    return wrap_setattr('_MODEL_NAME', value)

def setdatasetname(value):
    return wrap_setattr('_DG_NAME', value)


def IoU_arg_setting(func):
    def decorated(*args, **kwargs):
        
        ori_interpreter, ori_lambda_for_final = args[3].interpreter, args[3].lambda_for_final
        args[3].interpreter, args[3].lambda_for_final = 'lrp', 1
        
        ori_loss_type = args[3].loss_type
        args[3].loss_type = 'None'
        
        ori_mask_data_size = args[3].mask_data_size
        args[3].mask_data_size = 224
        
        ori_R_process = args[3].R_process
        args[3].R_process = None # None
        
        func(*args, **kwargs)
        
        args[3].interpreter = ori_interpreter
        args[3].lambda_for_final = ori_lambda_for_final
        args[3].loss_type = ori_loss_type
        args[3].mask_data_size = ori_mask_data_size
        args[3].R_process = ori_R_process
        
        return func
    return decorated

# def boundary(func):
#     def decorated(*args, **kwargs):
#         print('######################')
#         func(*args, **kwargs)
#         print('######################')
#         return func
#     return decorated

    
# @boundary
# def say_hello(name):
#     print('hello!! {}'.format(name))
    
# say_hello('juna')
# say_hello('jane')










































