import matplotlib.pyplot as plt
import pickle
import os

fault_folder = './logs/normal'
client_num = 4
fault_index = -1

# plt.suptitle("normal_loss")
# for i in range(client_num):
#     with open(os.path.join(fault_folder, f'history-{i}-fault-{fault_index}.pkl'), 'rb') as f:
#         history = pickle.load(f)
#     plt.subplot(2, 2, i+1)
#     ep = len(history['train']['loss'])
#     print(ep)
#     print(history['train']['loss'])
#     plt.ylim(0, 3)
#     plt.plot(range(ep), history['train']['loss'], label='Training Loss')
#     plt.plot(range(len(history['test']['loss'][:-1])), history['test']['loss'][:-1], label='Test Loss')
#     plt.legend()
#     plt.xlabel('Communication Rounds')
#     plt.ylabel('Loss')
#     plt.title(f'Client-{i} Loss')

# plt.tight_layout()
# plt.savefig("./img/normal_loss.png")
# plt.show()



# plt.suptitle("normal_acc")
# for i in range(client_num):
#     with open(os.path.join(fault_folder, f'history-{i}-fault-{fault_index}.pkl'), 'rb') as f:
#         history = pickle.load(f)
#     plt.subplot(2, 2, i+1)
#     ep = len(history['train']['acc'])
    
#     plt.ylim(0.5, 1)
#     plt.plot(range(ep), history['train']['acc'], label='Training Acc')
#     plt.plot(range(len(history['test']['acc'][:-1])), history['test']['acc'][:-1], label='Test Acc')
#     plt.legend()
#     plt.xlabel('Communication Rounds')
#     plt.ylabel('Accuracy')
#     plt.title(f'Client-{i} Accuracy')

# plt.tight_layout() 
# plt.savefig("./img/normal_acc.png")
# plt.show()



# #################################################################################################################


# fault_folder = './logs/with_attack'
# client_num = 4
# fault_index = 0

# plt.suptitle("attack_loss")

# for i in range(client_num):
#     with open(os.path.join(fault_folder, f'history-{i}-fault-{fault_index}.pkl'), 'rb') as f:
#         history = pickle.load(f)
    
#     plt.subplot(2, 2, i+1)
#     ep = len(history['train']['loss'])
#     print(ep)
#     print(history['train']['loss'])
#     plt.ylim(0, 3)
#     plt.plot(range(ep), history['train']['loss'], label='Training Loss')
#     plt.plot(range(len(history['test']['loss'][:-1])), history['test']['loss'][:-1], label='Test Loss')
#     plt.legend()
#     plt.xlabel('Communication Rounds')
#     plt.ylabel('Loss')
#     plt.title(f'Client-{i} Loss')

# plt.tight_layout()
# plt.savefig("./img/attack_loss.png")
# plt.show()



# plt.suptitle("attack_acc")

# for i in range(client_num):
#     with open(os.path.join(fault_folder, f'history-{i}-fault-{fault_index}.pkl'), 'rb') as f:
#         history = pickle.load(f)
#     plt.subplot(2, 2, i+1)
#     ep = len(history['train']['acc'])
    
#     plt.ylim(0.5, 1)
#     plt.plot(range(ep), history['train']['acc'], label='Training Acc')
#     plt.plot(range(len(history['test']['acc'][:-1])), history['test']['acc'][:-1], label='Test Acc')
#     plt.legend()
#     plt.xlabel('Communication Rounds')
#     plt.ylabel('Accuracy')
#     plt.title(f'Client-{i} Accuracy')

# plt.tight_layout() 
# plt.savefig("./img/attack_acc.png")
# plt.show()



# #################################################################################################################


exp_name = "1_attack_80_error_centralized_fashion"

plt.suptitle(f"{exp_name}_loss")
normal_folder = f'./logs/{exp_name}'
fault_index = 1
for i in range(client_num):
    with open(os.path.join(normal_folder, f'history-{i}-fault-{fault_index}.pkl'), 'rb') as f:
        history = pickle.load(f)
    
    plt.subplot(2, 2, i+1)
    ep = len(history['train']['loss'])
    # plt.ylim(0, 3)
    plt.plot(range(ep), history['train']['loss'], label='Training Loss')
    plt.plot(range(len(history['test']['loss'][:-1])), history['test']['loss'][:-1], label='Test Loss')
    plt.legend()
    plt.xlabel('Communication Rounds')
    plt.ylabel('Loss')
    plt.title(f'Client-{i} Loss')
plt.tight_layout() 
plt.savefig(f"./img/{exp_name}_loss.png")
plt.show()


plt.suptitle(f"{exp_name}_acc")
for i in range(client_num):
    with open(os.path.join(normal_folder, f'history-{i}-fault-{fault_index}.pkl'), 'rb') as f:
        history = pickle.load(f)
    plt.subplot(2, 2, i+1)
    ep = len(history['train']['acc'])
    # plt.ylim(0.5, 1)
    plt.plot(range(ep), history['train']['acc'], label='Training Acc')
    plt.plot(range(len(history['test']['acc'][:-1])), history['test']['acc'][:-1], label='Test Acc')
    plt.legend()
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy')
    plt.title(f'Client-{i} Accuracy')

plt.tight_layout() 
plt.savefig(f"./img/{exp_name}_acc.png")
plt.show()


