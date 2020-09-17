#pragma once

enum arg_decl_type {
    // Az argumentum egy sima string
    // Az outparam valodi tipusa `char const**`
    ARG_STRING,
    // Az argumentum egy elojeles egesz
    // Az outparam valodi tipusa `long*`
    ARG_LONG,
};

// Argumentum feldolgozo
struct arg_decl {
    char const *flag;
    void *outparam;
    arg_decl_type type;
};

static bool parse_args(
        int argc,
        char **argv,
        arg_decl const *args) {
    bool ret = true;

    for(int i = 1; i < argc; i++) {
        arg_decl const *ad = &args[0];
        while(ret && ad->flag != NULL) {
            if(strcmp(argv[i], ad->flag) == 0) {
                if(i + 1 < argc) {
                    switch(ad->type) {
                        case ARG_STRING:
                            *((char const**)ad->outparam) = argv[i + 1];
                            break;
                        case ARG_LONG:
                            ret &= sscanf(argv[i + 1], "%ld", (long*)ad->outparam);
                            break;
                    }
                    i++;
                    break;
                } else {
                    ret = false;
                }
            }

            ad++;
        }

        if(ad->flag == NULL) {
            ret = false;
        }
    }

    return ret;
}
