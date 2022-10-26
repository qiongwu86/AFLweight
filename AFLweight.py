from helper_function.utils import args_parser
import copy
args = args_parser()




def asy_average_weights(vehicle_idx, global_model, local_model, gamma):
    print("vehicle ", vehicle_idx+1, " has already updated the RSU!")


    for name, param in global_model.named_parameters():
        for name2, param2 in local_model.named_parameters():
            if name == name2:
                param.data.copy_(gamma * param.data + (1 - gamma) * param2.data)

    return global_model, global_model.state_dict()

def asy_average_weights_weight(vehicle_idx, global_model, local_model, gamma, local_param1, local_param2, local_param3):
    print("vehicle ", vehicle_idx + 1, " has already updated the RSU!")

    for name, param in global_model.named_parameters():
        for name2, param2 in local_model.named_parameters():
            if name == name2:
                param.data.copy_(gamma * param.data + (1 - gamma) * local_param1 * local_param2 * local_param3 * param2.data)

    return global_model, global_model.state_dict()



