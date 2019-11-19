import torch
import dqn


def extract_tensors(experiences):
	# Convert batch of Experiences to Experience of batches
	batch = dqn.Experience(*zip(*experiences))

	t1 = torch.cat(batch.state)
	t2 = torch.cat(batch.action)
	t3 = torch.cat(batch.reward)
	t4 = torch.cat(batch.next_state)

	return (t1,t2,t3,t4)

class QValues():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	@staticmethod
	def get_current(policy_net, states, actions):
		#States tensoru 512 uzunlukta. Actions tensoru 256. States'deki her ikiliye (x,y), actions'da bir action dusuyor.
		from_net = policy_net(states)
		#print(states.size())
		#print(actions.size())
		print(states)
		print("\n\n")
		print(actions)
		print("From net: " + str(from_net))
		return from_net.gather(dim=-1, index=actions.unsqueeze(-1))

	@staticmethod        
	def get_next(target_net, next_states):                
	    final_state_locations = next_states.flatten(start_dim=1) \
	        .max(dim=1)[0].eq(0).type(torch.bool)
	    non_final_state_locations = (final_state_locations == False)
	    non_final_states = next_states[non_final_state_locations]
	    batch_size = next_states.shape[0]
	    values = torch.zeros(batch_size).to(QValues.device)
	    values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
	    return values