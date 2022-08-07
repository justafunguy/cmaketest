import os
import sys
import onnx
import cv2
import numpy as np
import subprocess
import shlex
import struct

def get_model_input(model):
    input_all = [node.name for node in model.graph.input]
    input_initializer =  [node.name for node in model.graph.initializer]
    input_names = list(set(input_all)  - set(input_initializer))
    inputs = []
    for x in model.graph.input:
        if x.name in input_names:
            inputs.append(x)
    return inputs

def create_rand_feature(dim, max_val, dtype="int16"):
    size = 1
    while len(dim) < 4:
        dim.insert(0, 1)
    for i in range(0,4):
        size *= dim[i]
    feature = np.random.rand(size)
    feature = feature * max_val
    feature = feature.reshape(dim[0], dim[1], dim[2], dim[3])
    feature = feature.astype(dtype)
    return feature

def create_input_features(input, model_name, elem_type="float32", image_base_path = "../sample_nn_diag/images/"):
    image_base_path = os.path.join(image_base_path, model_name)
    if not os.path.exists(image_base_path):
        os.makedirs(image_base_path)
    
    verify_dir = os.path.join(image_base_path, "dra")
    quantity_dir = os.path.join(image_base_path, "test") 
    if not os.path.exists(verify_dir):
        os.makedirs(verify_dir)
    if not os.path.exists(quantity_dir):
        os.makedirs(quantity_dir)
    os.system("rm -rf " + verify_dir + "/*")
    os.system("rm -rf " + quantity_dir + "/*")

    verify_path = ""
    quantity_path = ""
    is_first_layer = True

    cvt_a = (6.103888154029846e-05, 6.103888154029846e-05, 6.103888154029846e-05, 6.103888154029846e-05)
    cvt_b =(-0.06377648562192917, -0.06366174668073654, -0.06414605677127838, -0.06403859704732895)
    cvt_c =(16383.0, 16383.0, 16383.0, 16383.0)
    cvt_d =(1044.8502197265625, 1042.970458984375, 1050.9049072265625, 1049.144287109375)
    cvt_a = np.array(cvt_a).astype("float32")
    cvt_b = np.array(cvt_b).astype("float32")
    cvt_c = np.array(cvt_c).astype("float32")
    cvt_d = np.array(cvt_d).astype("float32")

    for x in input:
        layer_name = x.name
        if not os.path.exists(layer_name):
            os.system("mkdir " + layer_name)
        dim = []
        for y in x.type.tensor_type.shape.dim:
            dim.append(y.dim_value)

        if elem_type in ["int16","uint16"]:
            maxnum = 2**15
        elif elem_type in ["int8", "uint8"]:
            maxnum = 2**7
        else:
            maxnum = 1
        for i in range(1):
            if len(dim) == 4:
                if dim[0] == 1 and dim[1] == 3:
                    feature = create_rand_feature(dim, 255, "uint8")
                    feature = feature.reshape(dim[2], dim[3], dim[1])
                    cv2.imwrite(os.path.join(layer_name, str(i) + ".jpg"), feature)
                elif layer_name.find("cvt_a") == 0:
                    cvt_a.tofile(os.path.join(layer_name, str(i) + ".bin"))
                elif layer_name.find("cvt_b") == 0:
                    cvt_b.tofile(os.path.join(layer_name, str(i) + ".bin"))
                elif layer_name.find("cvt_c") == 0:
                    cvt_c.tofile(os.path.join(layer_name, str(i) + ".bin"))
                elif layer_name.find("cvt_d") == 0:
                    cvt_d.tofile(os.path.join(layer_name, str(i) + ".bin"))
                else:
                    feature = create_rand_feature(dim, maxnum, elem_type)
                    feature.tofile(os.path.join(layer_name, str(i) + ".bin"))
            else:
                feature = create_rand_feature(dim, maxnum, elem_type)
                feature.tofile(os.path.join(layer_name, str(i) + ".bin"))


        cmd = "mv " + layer_name + " " + os.path.join(image_base_path, "dra")
        os.system(cmd)

        if is_first_layer:
            verify_path += "USR_TEST_INPUT_DRA_DIR        = @srcdir@/../../../images/{}/dra/".format(model_name) + layer_name + "\n"
            quantity_path += "USR_TEST_INPUT_DIR            = @srcdir@/../../../images/{}/test/".format(model_name) + layer_name + "\n"
            is_first_layer = False
        else:
            verify_path += "USR_TEST_INPUT_DRA_DIR       += @srcdir@/../../../images/{}/dra/".format(model_name) + layer_name + "\n"
            quantity_path += "USR_TEST_INPUT_DIR           += @srcdir@/../../../images/{}/test/".format(model_name) + layer_name + "\n"

    cmd = "cp -r " + os.path.join(image_base_path, "dra/*") + " " + os.path.join(image_base_path, "test")
    os.system(cmd)

    return verify_path, quantity_path


def get_layer_makein_info(layer, is_input = True):
    # dims = []
    layer_dims = ""
    layer_name = ""
    out_df_cfg = ""
    out_df = "1,2,0,7"
    is_first_layer = True
    prefix = ["","",""]
    if (is_input):
        prefix[0] += "USR_TEST_INPUT_DIM           "
        prefix[1] += "USR_TEST_INPUT_LAYER         "
    else:
        prefix[0] += "USR_TEST_FORCE_OUT_SHAPE     "
        prefix[1] += "USR_TEST_OUT_LAYER           "
        prefix[2] += "USR_TEST_FORCE_OUT_DF        "
    for x in layer:
        if is_first_layer:
            layer_dims += prefix[0] + " = "
            layer_name += prefix[1] + " = " + x.name
            out_df_cfg += prefix[2] + " = " + out_df
            is_first_layer = False
        else:
            layer_dims += prefix[0] + "+= "
            layer_name += prefix[1] + "+= " + x.name
            out_df_cfg += prefix[2] + "+= " + out_df
        layer_name += "\n"
        out_df_cfg += "\n"
        is_first_dim = True
        cur_dim = ""
        for y in x.type.tensor_type.shape.dim:
            if is_first_dim:
                cur_dim += str(y.dim_value)
                is_first_dim = False
            else:
                cur_dim += "," + str(y.dim_value)
        
        cur_len = len(cur_dim.split(","))
        for _ in range(cur_len, 4):
            cur_dim = "1," + cur_dim
        layer_dims += cur_dim + "\n"
    if not is_input:
        layer_dims += out_df_cfg
    return layer_name, layer_dims

def get_input_format(layer, elem_type):
    input_df = ""
    input_cf = ""
    is_first_layer = True
    prefix = ["",""]
    prefix[0] += "USR_TEST_INPUT_DF            "
    prefix[1] += "USR_TEST_INPUT_CF            "
    for x in layer:
        print(x)
        if x.type.tensor_type.shape.dim[0].dim_value == 1 and x.type.tensor_type.shape.dim[1].dim_value == 3:
            if is_first_layer:
                input_df += prefix[0] + " = 0,0,0,0"
                input_cf += prefix[1] + " = 0"
                is_first_layer = False
            else:
                input_df += prefix[0] + "+= 0,0,0,0"
                input_cf += prefix[1] + "+= 0"
        elif x.name.find("cvt") != -1 or x.name.find("fusion") != -1:
            if is_first_layer:
                input_df += prefix[0] + " = 1,2,0,7"
                input_cf += prefix[1] + " = 2"
                is_first_layer = False
            else:
                input_df += prefix[0] + "+= 1,2,0,7"
                input_cf += prefix[1] + "+= 2"
        else:
            if elem_type == "float32":
                if is_first_layer:
                    input_df += prefix[0] + " = 1,2,0,7"
                    input_cf += prefix[1] + " = 2"
                    is_first_layer = False
                else:
                    input_df += prefix[0] + "+= 1,2,0,7"
                    input_cf += prefix[1] + "+= 2"
            elif elem_type == "uint8":
                if is_first_layer:
                    input_df += prefix[0] + " = 0,0,0,0"
                    input_cf += prefix[1] + " = 0"
                    is_first_layer = False
                else:
                    input_df += prefix[0] + "+= 0,0,0,0"
                    input_cf += prefix[1] + "+= 0"
            elif elem_type == "int8":
                if is_first_layer:
                    input_df += prefix[0] + " = 1,0,0,0"
                    input_cf += prefix[1] + " = 2"
                    is_first_layer = False
                else:
                    input_df += prefix[0] + "+= 1,0,0,0"
                    input_cf += prefix[1] + "+= 2"
            elif elem_type == "int16":
                if is_first_layer:
                    input_df += prefix[0] + " = 1,1,0,0"
                    input_cf += prefix[1] + " = 2"
                    is_first_layer = False
                else:
                    input_df += prefix[0] + "+= 1,1,0,0"
                    input_cf += prefix[1] + "+= 2"
            elif elem_type == "uint16":
                if is_first_layer:
                    input_df += prefix[0] + " = 0,1,0,0"
                    input_cf += prefix[1] + " = 2"
                    is_first_layer = False
                else:
                    input_df += prefix[0] + "+= 0,1,0,0"
                    input_cf += prefix[1] + "+= 2"
            elif elem_type == "float16":
                if is_first_layer:
                    input_df += prefix[0] + " = 1,2,0,4"
                    input_cf += prefix[1] + " = 2"
                    is_first_layer = False
                else:
                    input_df += prefix[0] + "+= 1,2,0,4"
                    input_cf += prefix[1] + "+= 2"
            else:
                print("error: unsupported data type")
                exit(0)

        input_df += "\n"
        input_cf += "\n"
    return input_df, input_cf

def get_layer_info(layer):
    layer_dims = "{"
    layer_name = "{"
    is_first_layer = True
    for x in layer:
        if is_first_layer:
            layer_dims += "{"
            layer_name += "\"" + x.name + "\""
            is_first_layer = False
        else:
            layer_dims += ", {"
            layer_name += ", \"" + x.name + "\""
        is_first_dim = True
        for y in x.type.tensor_type.shape.dim:
            if is_first_dim:
                layer_dims += str(y.dim_value)
                is_first_dim = False
            else:
                layer_dims += ", " + str(y.dim_value)
        layer_dims += "}"

    layer_dims += "}"
    layer_name += "}"

    return layer_name, layer_dims


def create_input_config(input, model_name, elem_size = 1, align_size = 128):
    input_num = len(input)
    if not os.path.exists("rtos_out"):
        os.makedirs("rtos_out")
    with open(os.path.join("rtos_out", model_name + ".cfg"), "wb") as f:
        s = struct.pack("i", input_num)
        f.write(s)
        for x in input:
            print(x.name)
            print(x.type.tensor_type.shape.dim)
            if(len(x.type.tensor_type.shape.dim)!=4):
                print("warning: input dim is not 4")
                break
            for i in range(4):
                dim = x.type.tensor_type.shape.dim[i].dim_value
                s = struct.pack("i", dim)
                f.write(s)

            # dtype = 2
            # if x.name.find("cvt") != -1 or x.name.find("fusion") != -1:
            #     dtype = 4
            s = struct.pack("i", elem_size)
            f.write(s)

def calc_cvt_ab(k, sigma2, anchor_k, anchor_sigma2, black_levels, white_level,
                use_ksigma, data_scale):
    black_levels = torch.tensor(black_levels).view(1, -1, 1, 1).float()
    white_level = torch.tensor(white_level).view(1, 1, 1, 1).float()
    if use_ksigma:
        k = torch.tensor(k).view(1, 4, 1, 1).float()
        sigma2 = torch.tensor(sigma2).view(1, 4, 1, 1).float()
        anchor_k = torch.tensor(anchor_k).view(1, 4, 1, 1)
        anchor_sigma2 = torch.tensor(anchor_sigma2).view(1, 4, 1, 1)
        cvt_a = anchor_k / k
        cvt_b = -black_levels / k * anchor_k + sigma2 / k**2 * anchor_k - anchor_sigma2 / anchor_k
    else:
        cvt_a = 1
        cvt_b = -black_levels
    cvt_a *= data_scale / white_level
    cvt_b *= data_scale / white_level
    return cvt_a, cvt_b

    
if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("[CVT_ERR] USAGE: python3 convert_model.py model[*.onnx] | coverage_th[0-1] | precision[0/1/2] | elem_type [int16,uint16,int8,uint8,float16,float32] | batch2channel[0/1]")
        exit(0)

    supported_datatype = ["int16", "uint16", "int8", "uint8", "float16", "float32"]
    # parse param
    model_path = sys.argv[1]
    print("[CVT_INFO] model_path: " + model_path)
    coverage_th = sys.argv[2]
    print("[CVT_INFO] coverage_th: " + coverage_th)
    if (sys.argv[3] == "0"):
        precision = "mix"
        dra_opt_act_force = ""
        dra_opt_coeff_force = ""
        print("[CVT_INFO] use mixed quant")
    elif (sys.argv[3] == "1"):
        precision = "fx8"
        dra_opt_act_force = ",act-force-fx8"
        dra_opt_coeff_force = ",coeff-force-fx8"
        print("[CVT_INFO] use force-fx8 quant")
    elif (sys.argv[3] == "2"):
        precision = "fx16"
        dra_opt_act_force = ",act-force-fx16"
        dra_opt_coeff_force = ",coeff-force-fx16"
        print("[CVT_INFO] use force-fx16 quant")
    else:
        print("[CVT_ERR] invalid precision type")
        exit(0)
    elem_type = (sys.argv[4])
    if elem_type not in supported_datatype:
        print("[CVT_ERROR] unsupported data type {}".format(elem_type))
        exit(0)
    batch2channel = (int)(sys.argv[5])

    tar_profiler_model_dir = "../sample_nn_diag/cnn_model/profiler"
    if not os.path.exists(tar_profiler_model_dir):
        os.makedirs(tar_profiler_model_dir)
    os.system("cp " + model_path + " " + tar_profiler_model_dir)

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_model_name = model_name #+ "_profiler_th" + coverage_th + "_" + precision

    model = onnx.load(model_path)
    # onnx.checker.check_model(model)
    input = get_model_input(model)
    output = model.graph.output

    if input[0].type.tensor_type.shape.dim[0].dim_value > 1 or batch2channel == 1:
        print("[CVT_INFO] run batch to channel >>>>>>")
        work_dir = "../rtos/cortex_a"
        cmd = "source /opt/amba/env/env_set.sh cv5 /opt/amba"

        pwd = os.getcwd()
        model_abs_path = os.path.join(pwd, model_path)
        btc_model_abs_path = model_abs_path.replace(".onnx","_btc.onnx")
        cmd += " && graph_surgery.py onnx -m {} -t BatchToChannel -o {} ".format(model_abs_path, btc_model_abs_path)
        print("[CVT_CMD] "+cmd)
        p = subprocess.Popen(['/bin/bash', '-c', cmd], cwd = work_dir)
        ret = p.wait()
        if ret != 0:
            exit(ret)

        os.system("cp " + btc_model_abs_path + " " + tar_profiler_model_dir + "/" + os.path.basename(model_path))
        model = onnx.load(btc_model_abs_path)
        input = get_model_input(model)
        output = model.graph.output

    elem_size = 1
    if elem_type in ["float16", "int16", "uint16"]:
        elem_size = 2
    elif elem_type in ["int8", "unint8"]:
        elem_size = 1
    else:
        elem_size = 4
    create_input_config(input, output_model_name, elem_size=elem_size)

    # network_name = "profiler"
    network_name = model_name[0:22]#The maximum length is 22 characters.

    makefile_cfg = """
USR_ADK_PATH                  = @srcdir@/../../../../adk
USR_NETWORK                   = {0}
USR_PARSER_ENV                = onnx
USR_ARM_ENV                   = rtos
USR_CAFFE_PROTOTXT_FILE       = 
USR_CAFFE_MODEL_FILE          = 
USR_TF_PROTOBUF_FILE          = 
USR_ONNX_FILE                 = @srcdir@/../../../cnn_model/profiler/{1}.onnx
USR_VPP_DIR                   = 
USR_JSON_PREPOST_PROC_FILE    = 
USR_DBG_FLAG                  = 
USR_GRAPH_SURGERY_FLAG        = 
USR_GRAPH_SURGERY_OPT         = 
USR_GRAPH_SURGERY_INPUT_LAYER = 
USR_GRAPH_SURGERY_INPUT_DIM   = 
USR_GRAPH_SURGERY_OUT_LAYER   = 
USR_DRA_MODE                  = 2
USR_DRA_OPT                   = coverage_th={2}{3}{4}
USR_PARSER_GENERAL_OPT        = 
USR_CN_DIR                    = 
USR_CN_PARSER_FN              = 
USR_CN_DLL_FN                 = 
USR_CN_COMPILE_FLAG           = 
USR_VAS_OPT                   = 
""".format(network_name, model_name, coverage_th, dra_opt_act_force, dra_opt_coeff_force)

    verify_path, quantity_path = create_input_features(input, model_name, elem_type)
    input_name, input_dims = get_layer_makein_info(input, True)

    makefile_cfg += verify_path
    makefile_cfg += input_name
    makefile_cfg += quantity_path

    input_df, input_cf = get_input_format(input, elem_type)
    makefile_cfg += input_df
    makefile_cfg += input_cf

    makefile_cfg += """
USR_TEST_INPUT_MEAN           = 
USR_TEST_INPUT_SCALE          = 
"""
    makefile_cfg += input_dims

    output_name, output_dims = get_layer_makein_info(output, False)
    makefile_cfg += output_name
    makefile_cfg += output_dims

    makefile_cfg += """
USR_TEST_FORCE_OUT_TRANSPOSE  = 
USR_BUB_FWK_DIR               = @srcdir@/../../../../rtos/cortex_a/svc/comsvc/cv
USR_INPUT_YUV420_FLAG         = 0
USR_OUTPUT_FP32_FLAG          =
USR_BATCH_SIZE                = 
USR_SUPERDAG_TYPE             = 5
USR_HL_DISPLAY_NAME           = 
USR_IDSP_ROI_SCALE            = 
USR_IDSP_ROI_X                = 
USR_IDSP_ROI_Y                = 
USR_LNX_ARM_APP               = 
USR_LNX_BUILD_DIR             = 

define cnn_preprocess
    @echo "no preprocess actions !!"
endef

define cnn_postprocess
endef

include Makefile.internal_cfg
"""

    f = open("Makefile.in", "w")
    f.write(makefile_cfg)
    f.close()
    print("\nMakefile.in:")
    print(makefile_cfg)
    print("============================================================\n")

    profiler_cfg_dir = "../sample_nn_diag/diags/file_in/01100_{}/".format(model_name)
    if not os.path.exists(profiler_cfg_dir):
        os.makedirs(profiler_cfg_dir)
    
    cmd = "cp ../sample_nn_diag/diags/file_in/00000_lenet_cf/* " + profiler_cfg_dir
    print("[CVT_CMD] "+cmd)
    os.system(cmd)

    cmd = "mv Makefile.in " + profiler_cfg_dir
    print("[CVT_CMD] "+cmd)
    os.system(cmd)

    cmd = "cp ./.diags.mk ../rtos/cortex_a/svc/comsvc/cv/diags.mk"
    print("[CVT_CMD] "+cmd)
    os.system(cmd)

    work_dir = "../rtos/cortex_a"

    test_dir = os.path.join(work_dir, model_name)
    if os.path.exists(test_dir):
        cmd = "rm -rf " + test_dir
        print("[CVT_CMD] "+cmd)
        os.system(cmd)

    os.makedirs(test_dir)

    print("\n")
    cmd = "source /opt/amba/env/env_set.sh cv5 /opt/amba"

    pwd = os.getcwd()
    file_in_dir = os.path.join(pwd, profiler_cfg_dir)
    cmd += " && cd {0} && remoteconfig ".format(model_name) + file_in_dir + " && make && cd .."
    # cmd += " && source /opt/amba/env/env_set.sh cv5 /opt/amba && make diags CROSS_COMPILE=aarch64-none-elf- "
    print("[CVT_CMD] "+cmd)
    p = subprocess.Popen(['/bin/bash', '-c', cmd], cwd = work_dir)
    ret = p.wait()
    if ret != 0:
        exit(ret)

    cmd = "source /opt/amba/env/env_set.sh cv5 /opt/amba && make diags CROSS_COMPILE=aarch64-none-elf- "
    print("[CVT_CMD] "+cmd)
    p = subprocess.Popen(['/bin/bash', '-c', cmd], cwd = work_dir)
    ret = p.wait()
    if ret != 0:
        exit(ret)

    outdir = "./rtos_out"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    cmd = "cp -r ../rtos/cortex_a/output/out/cv/{0}_ag/flexibin/flexibin0.bin {1}/{2}.bin".format(network_name, outdir, model_name)
    print("[CVT_CMD] "+cmd)
    os.system(cmd)

