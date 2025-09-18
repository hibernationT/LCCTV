from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.antiuav_path = ''
    settings.davis_dir = ''
    settings.got10k_path = '/media/sot_datasets/GOT10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = '/media/sot_datasets/LaSOT'
    settings.lasot_extension_subset_path = '/media/sot_datasets/LaSOT_extension_subset'
    settings.network_path = ''    # Where tracking networks are stored.
    settings.nfs_path = '/media/sot_datasets/NFS30'
    settings.otb_path = '/media/sot_datasets/OTB100'
    settings.my_path = '/media/sot_datasets/mydata'
    settings.result_plot_path = ''
    settings.results_path = ''    # Where to store tracking results
    settings.segmentation_path = ''
    settings.prj_dir = ''
    settings.save_dir = ''
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/media/sot_datasets/TrackingNet'
    settings.uav_path = '/media/sot_datasets/UAV123'
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

