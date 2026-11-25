function model = process_0_create()

model.type = 'model';
model.method = 'method_0';


model.processing.background = [];
model.processing.gauss_filter = 1;
model.processing.median_filter = 3;
model.processing.boundary_threshold = 1;  % to be defined by user
model.processing.min_area = 30;
model.processing.max_area = 5000;
model.processing.roi = [];
