import torch
import dqn

def extract_tensors(experiences):
	# Convert batch of Experiences to Experience of batches
	batch = dqn.Experience(*zip(*experiences))
	state_batch = []
	nextState_batch = []

	for d in experiences:
		#for state batch
		a = d[0].cpu().detach().numpy()
		a = [a]
		state_batch += a
		#for nextState batch
		b = d[2].cpu().detach().numpy()
		b = [b]
		nextState_batch += b

	state_batch = torch.tensor(state_batch, dtype=torch.float32)
	action_batch = torch.cat(tuple(d[1] for d in experiences))
	nextState_batch = torch.tensor(nextState_batch, dtype=torch.float32)
	reward_batch = torch.cat(tuple(d[3] for d in experiences))

	return (state_batch,action_batch,reward_batch,nextState_batch)



