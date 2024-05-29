/* Tracting the function call, ret and function instruction
 *
 *
 *
 */

#include "dr_api.h"
#include "drcovlib.h"
#include "drmgr.h"
#ifdef SHOW_SYMBOLS
#include "drsyms.h"
#endif
#include "utils.h"
// #include <string>


static void
event_exit(void);
static void
event_thread_init(void *drcontext);
static void
event_thread_exit(void *drcontext);
static dr_emit_flags_t
event_bb_insert(void *drcontext, void *tag, instrlist_t *bb, instr_t *instr,
                bool for_trace, bool translating, void *user_data);

static client_id_t my_id;
static int tls_idx;
// static char *log_func;

DR_EXPORT void dr_client_main(client_id_t id, int argc, const char *argv[])
{
    drcovlib_options_t ops ={sizeof(ops), 1};

    //options_init( id , argc, argv, &ops);

    if(drcovlib_init(&ops) != DRCOVLIB_SUCCESS){
        //NOTIFY(0, "fetal error:drcovlib failed to initialize\n");
        dr_abort();
    }

    drmgr_init();
    my_id = id;
    /* make it easy to tell, by looking at log file, which client executed */
    // dr_log(NULL, DR_LOG_ALL, 2, "Testting......\n");
    dr_log(NULL, DR_LOG_ALL, 1, "Client 'tract_func' initializing\n");

/* also give notification to stderr */
#ifdef SHOW_RESULTS
    if (dr_is_notify_on())
    {
#ifdef WINDOWS
        /* ask for best-effort printing to cmd window.  must be called at init. */
        dr_enable_console_printing();
#endif
        dr_fprintf(STDERR, "Client tractting is running\n");
    }
#endif

    /* register events */
    dr_register_exit_event(event_exit);
    drmgr_register_bb_instrumentation_event(NULL, event_bb_insert,
                                            NULL);
    // open recorde log file & close recorde log file
    drmgr_register_thread_init_event(event_thread_init);
    drmgr_register_thread_exit_event(event_thread_exit);

#ifdef SHOW_SYMBOLS
    if (drsym_init(0) != DRSYM_SUCCESS)
    {
        dr_log(NULL, DR_LOG_ALL, 1, "WARNING: unable to initialize symbol translation\n");
    }
#endif

    tls_idx = drmgr_register_tls_field();
    DR_ASSERT(tls_idx > -1);
}

static void
event_exit(void)
{
#ifdef SHOW_SYMBOLS
    if (drsym_exit() != DRSYM_SUCCESS)
    {
        dr_log(NULL, DR_LOG_ALL, 1, "WARNING: error cleaning up symbol library\n");
    }
#endif
    drmgr_unregister_tls_field(tls_idx);
    drmgr_exit();
    drcovlib_exit();
}

#ifdef WINDOWS
#define IF_WINDOWS(x) x
#else
#define IF_WINDOWS(x) /* nothing */
#endif

static void
event_thread_init(void *drcontext)
{
    file_t f;
    /* We're going to dump our data to a per-thread file.
     * On Windows we need an absolute path so we place it in
     * the same directory as our library. We could also pass
     * in a path as a client argument.
     */
    /// home/xiaoyu_yi/DynamoRIO-Linux-9.0.1/yxy_build/bin/libtract_func.so
    f = log_file_open(my_id, drcontext, NULL /* client lib path */, "tractFunc",
#ifndef WINDOWS
                      DR_FILE_CLOSE_ON_FORK |
#endif
                          DR_FILE_ALLOW_LARGE);
    DR_ASSERT(f != INVALID_FILE);

    /* store it in the slot provided in the drcontext */
    drmgr_set_tls_field(drcontext, tls_idx, (void *)(ptr_uint_t)f);
}

static void
event_thread_exit(void *drcontext)
{
    log_file_close((file_t)(ptr_uint_t)drmgr_get_tls_field(drcontext, tls_idx));
}

// recorde call and module(library) information
#ifdef SHOW_SYMBOLS
#define MAX_SYM_RESULT 256
static void
print_address(file_t f, app_pc addr, const char *prefix)
{
    drsym_error_t symres;
    drsym_info_t sym;
    char name[MAX_SYM_RESULT];
    char file[MAXIMUM_PATH];
    module_data_t *data;
    data = dr_lookup_module(addr);
    if (data == NULL)
    {
        dr_fprintf(f, "%s " PFX " ? ??:0\n", prefix, addr);
        return;
    }
    sym.struct_size = sizeof(sym);
    sym.name = name;
    sym.name_size = MAX_SYM_RESULT;
    sym.file = file;
    sym.file_size = MAXIMUM_PATH;
    symres = drsym_lookup_address(data->full_path, addr - data->start, &sym,
                                  DRSYM_DEFAULT_FLAGS);
    if (symres == DRSYM_SUCCESS || symres == DRSYM_ERROR_LINE_NOT_AVAILABLE)
    {
        const char *modname = dr_module_preferred_name(data);
        if (modname == NULL)
            modname = "<noname>";
        dr_fprintf(f, "%s " PFX " %s!%s+" PIFX, prefix, addr, modname, sym.name,
                   addr - data->start - sym.start_offs);
        if (symres == DRSYM_ERROR_LINE_NOT_AVAILABLE)
        {
            dr_fprintf(f, " ??:0\n");
        }
        else
        {
            dr_fprintf(f, " %s:%" UINT64_FORMAT_CODE "+" PIFX "\n", sym.file, sym.line,
                       sym.line_offs);
        }
    }
    else
        dr_fprintf(f, "%s " PFX " ? ??:0\n", prefix, addr);
    dr_free_module_data(data);
}
#endif

/*****call instruction of direct (problem)
 * 1) memory leak
 * 2) system call
 * 3) code coverage (for main program)
 * 4) thrid library (file system API)
 *
 *
 *
 *
 * */

static void
at_call(app_pc instr_addr, app_pc target_addr, uint call_type)
{
    // void *drcontext = dr_get_current_drcontext();
    file_t f =
        (file_t)(ptr_uint_t)drmgr_get_tls_field(dr_get_current_drcontext(), tls_idx);
    /*for tracting instrcution in bb*/
    dr_mcontext_t mc = {sizeof(mc), DR_MC_CONTROL /*only need xsp*/};
    dr_get_mcontext(dr_get_current_drcontext(), &mc);
#ifdef SHOW_SYMBOLS
    print_address(f, instr_addr, "CALL @ ");
    print_address(f, target_addr, "\t to ");
    dr_fprintf(f, "\tTOS is " PFX "\n", mc.xsp);
#else
    dr_fprintf(f, "CALL @ " PFX " to " PFX ", TOS is " PFX "\n", instr_addr, target_addr,
               mc.xsp);
#endif

    // void *drcontext = dr_get_current_drcontext();
    // PMyThreadData data = (PMyThreadData)drmgr_get_tls_field(drcontext, tls_idx);
    // instr_t *instr;
    // app_pc next_instr = 0;
    // instrlist_t *bb = decode_as_bb(drcontext, instr_addr);
    // instr = instrlist_last(bb);
    // app_pc pc = instr_get_app_pc(instr);
    // int len = instr_length(drcontext, instr);
    // next_instr = pc + len;
    // instrlist_clear_and_destroy(drcontext, bb);
    // // 添加记录
    // call_table_entry_add(drcontext, data, call_type, instr_addr, target_addr, next_instr,
    //                      0, (uint)(next_instr - instr_addr));
}

// 间接调用
static void
at_call_ind(app_pc instr_addr, app_pc target_addr)
{
    file_t f =
        (file_t)(ptr_uint_t)drmgr_get_tls_field(dr_get_current_drcontext(), tls_idx);

    /*for tracting instrcution in bb*/
    dr_mcontext_t mc = {sizeof(mc), DR_MC_CONTROL /*only need xsp*/};
    dr_get_mcontext(dr_get_current_drcontext(), &mc);
#ifdef SHOW_SYMBOLS
    print_address(f, instr_addr, "CALL @ ");
    print_address(f, target_addr, "\t to ");
    dr_fprintf(f, "\tTOS is " PFX "\n", mc.xsp);
#else
    dr_fprintf(f, "CALL @ " PFX " to " PFX ", TOS is " PFX "\n", instr_addr, target_addr,
               mc.xsp);
#endif
}

// 返回调用
static void
at_return(app_pc instr_addr, app_pc target_addr)
{
    // uint mod_id;
    // app_pc mod_start;
    // if (target_addr && (uint)target_addr < 0xBFFFFFFF)
    // {

    //     void *drcontext = dr_get_current_drcontext();
    //     dr_mcontext_t mc = {sizeof(mc), DR_MC_CONTROL /*only need xsp*/};
    //     PMyThreadData data = (PMyThreadData)drmgr_get_tls_field(drcontext, tls_idx);
    //     instrlist_t *bb = decode_as_bb(drcontext, instr_addr);
    //     instr = instrlist_last(bb);
    //     app_pc pc = instr_get_app_pc(instr);
    //     int len = instr_length(drcontext, instr);
    //     app_pc next_instr = pc + len;
    //     instrlist_clear_and_destroy(drcontext, bb);
    //     // 添加记录
    //     call_table_entry_add(drcontext, data, FUNC_RETURN, instr_addr, target_addr, 0,
    //                          0, (uint)(next_instr - instr_addr));
    // }
    file_t f =
        (file_t)(ptr_uint_t)drmgr_get_tls_field(dr_get_current_drcontext(), tls_idx);

    /*for tracting instrcution in bb*/
    dr_mcontext_t mc = {sizeof(mc), DR_MC_CONTROL /*only need xsp*/};
    dr_get_mcontext(dr_get_current_drcontext(), &mc);
#ifdef SHOW_SYMBOLS
    print_address(f, instr_addr, "RET @ ");
    print_address(f, target_addr, "\t to ");
    dr_fprintf(f, "\tTOS is " PFX "\n", mc.xsp);
#else
    dr_fprintf(f, "RET @ " PFX " to " PFX ", TOS is " PFX "\n", instr_addr, target_addr,
               mc.xsp);
#endif
}

static dr_emit_flags_t
event_bb_insert(void *drcontext, void *tag, instrlist_t *bb, instr_t *instr,
                bool for_trace, bool translating, void *user_data)
{
#ifdef VERBOSE
    file_t f =
        (file_t)(ptr_uint_t)drmgr_get_tls_field(dr_get_current_drcontext(), tls_idx);
    if (drmgr_is_first_instr(drcontext, instr))
    {
        // dr_printf("in dr_basic_block(tag=" PFX ")\n", tag);

        disassemble_set_syntax(DR_DISASM_INTEL);
        // instrlist_disassemble(drcontext, tag, bb, STDOUT);
        instrlist_disassemble(drcontext, tag, bb, f);
    }
#endif
    // identification different call and return instruction
    if (instr_is_cti(instr))
    {
        if (instr_is_call_direct(instr))
        {
            dr_insert_call_instrumentation(drcontext, bb, instr, (app_pc)at_call);
        }
        else if (instr_is_call_indirect(instr))
        {
            dr_insert_mbr_instrumentation(drcontext, bb, instr, (app_pc)at_call_ind,
                                          SPILL_SLOT_1);
        }
        else if (instr_is_return(instr))
        {
            dr_insert_mbr_instrumentation(drcontext, bb, instr, (app_pc)at_return,
                                          SPILL_SLOT_2);
        }
    }
    // 记录块间调用
    // app_pc tag_pc = dr_fragment_app_pc(tag);
    // dump_bb_inst_list_entry(drcontext, bb, data, tag_pc);
    return DR_EMIT_DEFAULT;
}
