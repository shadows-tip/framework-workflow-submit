from ModelPackage import Model

# Call propagation dynamics model for testing
model = Model(T=80, Nd=40000)
model.set_param(intervene_time=50, intervene_meas=1, intervene_param=[0.6])
res = model.get_result(label=['Ih'], plot=True)
