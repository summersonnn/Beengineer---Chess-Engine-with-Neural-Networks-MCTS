import torch
import dqn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_tensors(experiences):
	# Convert batch of Experiences to Experience of batches
	batch = dqn.Experience(*zip(*experiences))
	state_batch = []

	for d in experiences:
		a = d[0].cpu().detach().numpy()
		a = [a]
		#print(a)
		state_batch += a

	#print(state_batch)

	state_batch = torch.tensor(state_batch, dtype=torch.float)
	action_batch = torch.cat(tuple(d[1] for d in experiences))
	reward_batch = torch.cat(tuple(d[2] for d in experiences))
	nextState_batch = torch.cat(tuple(d[3] for d in experiences))

	return (state_batch,action_batch,reward_batch,nextState_batch)

class QValues():

	@staticmethod
	def get_current(policy_net, states, actions):
		return policy_net(states).gather(dim=-1, index=actions.unsqueeze(-1))


