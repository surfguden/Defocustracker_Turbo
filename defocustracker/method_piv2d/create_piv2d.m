function model = create_piv2d()

model.type = 'model';
model.method = 'piv2d';
model.processing.step = 1;
model.processing.IW1= [64 64];
model.processing.pitch = [32, 32];
model.processing.DX_minmax = [-16 16];
model.processing.DY_minmax = [-16 16];
model.processing.medfilt_kernel = 0;
model.processing.medfilt_replace_pxdiff = 1;
model.processing.bkg_image_mean = 0;
model.processing.roi = [];
