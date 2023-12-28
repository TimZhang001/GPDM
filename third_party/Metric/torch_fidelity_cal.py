import torch_fidelity

real_image_path = '/home/zhangss/PHDPaper/04_SinFusion_01/images/mvtec/wood/image'
fake_image_path = '/home/zhangss/PHDPaper/04_SinFusion_01/outputs/256_256_nextnet/wood'
metrics_dict = torch_fidelity.calculate_metrics(input1 = fake_image_path, 
                                                input2 = real_image_path, 
                                                cuda=True, 
                                                batch_size=8,
                                                isc=True, 
                                                fid=True, 
                                                kid=False,
                                                ppl=True, 
                                                prc=True, 
                                                verbose=True,)

print(metrics_dict)