#include <string.h>

#include "pesq.h"
#include "pesqio.h"
#include "pesqmain.h"

#ifdef _WIN32
#define PESQ_EXPORT __declspec(dllexport)
#else
#define PESQ_EXPORT
#endif

#define PESQ_SUCCESS PESQ_ERROR_SUCCESS

static void assign_name(char *dst_path, size_t dst_path_size, char *dst_file, size_t dst_file_size, const char *src) {
    strncpy(dst_path, src, dst_path_size - 1);
    dst_path[dst_path_size - 1] = '\0';
    strncpy(dst_file, src, dst_file_size - 1);
    dst_file[dst_file_size - 1] = '\0';
}

PESQ_EXPORT int pesq_backend(
    long sample_rate,
    const float *ref_data,
    int ref_len,
    const float *deg_data,
    int deg_len,
    int mode,
    float *out_score
) {
    long error_flag = 0;
    char *error_type = "unknown";

    SIGNAL_INFO ref_info;
    SIGNAL_INFO deg_info;
    ERROR_INFO err_info;

    memset(&ref_info, 0, sizeof(ref_info));
    memset(&deg_info, 0, sizeof(deg_info));
    memset(&err_info, 0, sizeof(err_info));

    select_rate(sample_rate, &error_flag, &error_type);
    if (error_flag != PESQ_SUCCESS) {
        return (int) error_flag;
    }

    assign_name(ref_info.path_name, sizeof(ref_info.path_name), ref_info.file_name, sizeof(ref_info.file_name), "reference-signal");
    assign_name(deg_info.path_name, sizeof(deg_info.path_name), deg_info.file_name, sizeof(deg_info.file_name), "degrade-signal");

    ref_info.Nsamples = ref_len;
    ref_info.apply_swap = 0;
    ref_info.input_filter = 1;
    ref_info.data = (float *) ref_data;

    deg_info.Nsamples = deg_len;
    deg_info.apply_swap = 0;
    deg_info.input_filter = 1;
    deg_info.data = (float *) deg_data;

    err_info.mode = NB_MODE;
    if (mode == 1) {
        ref_info.input_filter = 2;
        deg_info.input_filter = 2;
        err_info.mode = WB_MODE;
    }

    pesq_measure(&ref_info, &deg_info, &err_info, &error_flag, &error_type);
    if (error_flag != PESQ_SUCCESS) {
        return (int) error_flag;
    }

    if (out_score != NULL) {
        *out_score = err_info.mapped_mos;
    }
    return PESQ_SUCCESS;
}
