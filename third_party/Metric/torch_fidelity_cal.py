import torch_fidelity

real_image_path = '/home/zhangss/PHDPaper/06_ConSinGAN/TestIMG/Fake'
fake_image_path = '/home/zhangss/PHDPaper/06_ConSinGAN/TestIMG/True'
metrics_dict = torch_fidelity.calculate_metrics(input1 = fake_image_path, 
                                                input2 = real_image_path, 
                                                cuda=True, 
                                                batch_size=1,
                                                isc=False, 
                                                fid=True, 
                                                kid=True,
                                                ppl=False, 
                                                prc=False, 
                                                verbose=True,)

print(metrics_dict)