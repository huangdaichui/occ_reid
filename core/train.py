import torch
from tools import MultiItemAverageMeter, accuracy
from torchvision import transforms


def train_an_epoch(config, base, loaders, epoch=None):

	base.set_train()
	meter = MultiItemAverageMeter()

	### we assume 200 iterations as an epoch
	base.lr_scheduler.step(epoch)
	for _ in range(config.steps):

		### load a batch data
		imgs, pids, _ = loaders.train_iter.next_one()
		imgs, pids = imgs.to(base.device), pids.to(base.device)
		# toPIL = transforms.ToPILImage()
		# pic = toPIL(imgs[0, :, :, :])
		# pic.save('random.jpg')
		# toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
		#
		# pic = toPIL(imgs[1, :, :, :])
		# pic.save('random1.jpg')
		#
		# toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
		#
		# pic = toPIL(imgs[2, :, :, :])
		# pic.save('random3.jpg')
		# toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
		#
		# pic = toPIL(imgs[3, :, :, :])
		# pic.save('random4.jpg')

		if 'res' in config.cnnbackbone:
			### forward
			features, features_max, cls_score, result_list, local_align_result_list, cls_score_common, cls_score_max, cls_score_sum, ex_cl_result_list, ex_cl_local_align_list, ex_a_cl_result_list, ex_a_cl_local_align_list = base.model(imgs)
			# features, cls_score = base.model(imgs)

			### loss
			ide_loss = base.ide_creiteron(cls_score, pids)
			ide_loss_common = base.ide_creiteron(cls_score_common, pids)
			ide_loss_max = base.ide_creiteron(cls_score_max, pids)
			ide_loss_sum = base.ide_creiteron(cls_score_sum, pids)
			ide_local_loss = compute_local_loss(base.local_id_creiteron_list, result_list, pids)

			for i in range(13):
				if i == 0:
					local_align_loss = base.ide_creiteron(local_align_result_list[i], torch.full_like(pids, i).cuda())
				else:
					local_align_loss += base.ide_creiteron(local_align_result_list[i], torch.full_like(pids, i).cuda())

			for i in range(13):
				if i == 0:
					local_align_loss_ex = base.ide_creiteron(ex_cl_local_align_list[i], torch.full_like(pids, i).cuda())
				else:
					local_align_loss_ex += base.ide_creiteron(ex_cl_local_align_list[i], torch.full_like(pids, i).cuda())

			for i in range(13):
				if i == 0:
					local_align_loss_ex_a = base.ide_creiteron(ex_a_cl_local_align_list[i], torch.full_like(pids, i).cuda())
				else:
					local_align_loss_ex_a += base.ide_creiteron(ex_a_cl_local_align_list[i], torch.full_like(pids, i).cuda())

			#
			#
			#
			# # part_loss = part_loss_compute(features_list, pids)
			triplet_loss = base.triplet_creiteron(features, features, features, pids, pids, pids)
			triplet_loss_max = base.triplet_creiteron(features_max, features_max, features_max, pids, pids, pids)




			# for i in range(13):
			# 	triplet_loss += base.triplet_creiteron(local_align_result_list[i], local_align_result_list[i],
			# 										  local_align_result_list[i], pids, pids, pids)



			loss = ide_loss + triplet_loss + ide_loss_max + ide_loss_sum + triplet_loss_max \
				   + config.map_loss * ide_loss_common \
				   + config.local_loss * ide_local_loss \
				   + config.ch_loss * local_align_loss \
			       + 0.1 * (local_align_loss_ex_a + local_align_loss_ex)




			# loss = ide_loss + triplet_loss
			acc = accuracy(cls_score, pids, [1])[0]
			### optimize
			base.optimizer.zero_grad()
			loss.backward()
			base.optimizer.step()
			### recored
			meter.update({'ide_loss': ide_loss.data, 'triplet_loss': triplet_loss.data, 'acc': acc})
		elif config.cnnbackbone == 'osnetain':
			### forward
			if epoch < 10:
				cls_score = base.model(imgs, fixed_cnn=True)
			else:
				cls_score = base.model(imgs, fixed_cnn=False)
			### loss
			ide_loss = base.ide_creiteron(cls_score, pids)
			acc = accuracy(cls_score, pids, [1])[0]
			### optimize
			base.optimizer.zero_grad()
			ide_loss.backward()
			base.optimizer.step()
			### recored
			meter.update({'ide_loss': ide_loss.data, 'acc': acc})

	return meter.get_val(), meter.get_str()

def compute_local_loss(local_id_creiteron_list, result_list, pids):
	count1 = 0
	for local_id_creiteron, result in zip(local_id_creiteron_list, result_list):
		if count1 == 0:
			loss = local_id_creiteron(result, pids)
		else:
			loss += local_id_creiteron(result, pids)

	return loss

