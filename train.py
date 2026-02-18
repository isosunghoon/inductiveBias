from config import parse_args

try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

args, args_text = parse_args()

if args.log_wandb:
    if has_wandb:
        wandb.init(project=args.experiment, config=args)
    else: 
        print('ERROR: NO WANDB')

model = create_model()