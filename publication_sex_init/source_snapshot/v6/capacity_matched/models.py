"""Native model families with integer-matched, fully used parameter budgets."""
from torch import nn

from paper_odeae.models import PaperModel
from paper_odeae.baselines import BaselineModel


def field(state,widths):
    a,b=widths
    return nn.Sequential(nn.Linear(state,a),nn.Tanh(),nn.Linear(a,b),nn.Tanh(),nn.Linear(b,state))


def definition(dataset,kind):
    if dataset not in ['adni','mimic']:raise ValueError(dataset)
    if kind=='ode_ae':
        a,b,c,v=(247,247,246,244) if dataset=='adni' else (250,253,249,251)
        return {'latent':6,'mlp_width':a,'init_width':a,'update_width':b,'decoder_width':c,
                'ode_width':v,'encoder_ode_width':v,'generator_ode_width':v,
                'encoder_direction':'backward','encoder_initialization':'age','shared_vector_field':False}
    if kind=='ode_rnn':
        return {'latent':38,'encoder_hidden':38,'ode_width':88 if dataset=='adni' else 102,
                'generator_field_widths':[88,188] if dataset=='adni' else [102,162]}
    if kind=='latent_ode':
        return {'latent':32,'encoder_hidden':48,'ode_width':61 if dataset=='adni' else 60,
                'encoder_field_widths':[61,61] if dataset=='adni' else [60,61],
                'generator_field_widths':[53,59] if dataset=='adni' else [55,55]}
    raise ValueError(kind)


def parameter_count(dataset,kind):
    cfg=definition(dataset,kind);features=49 if dataset=='adni' else 44
    def mlp(a,b,w):return (a+1)*w+(w+1)*b
    def ode(d,widths):
        a,b=widths;return (d+1)*a+(a+1)*b+(b+1)*d
    if kind=='ode_ae':
        d=cfg['latent'];h=2*d
        return mlp(1,h,cfg['init_width'])+mlp(h+features,h,cfg['update_width'])+mlp(d,features,cfg['decoder_width'])+mlp(h,h,cfg['ode_width'])+mlp(d,d,cfg['ode_width'])
    inputs=features+(dataset=='mimic');d=cfg['latent']
    value=ode(d,cfg['generator_field_widths'])+(d+1)*features
    if kind=='ode_rnn':return value+3*d*(d+2*inputs+1)
    h=cfg['encoder_hidden']
    return value+ode(h,cfg['encoder_field_widths'])+3*h*(h+2*inputs+1)+(h+1)*h+(h+1)*2*d


def make_model(schema,cfg):
    kind=cfg['model']
    if kind=='ode_ae':
        return PaperModel(schema,kind='ode_ae',latent=cfg['latent'],ode_width=cfg['ode_width'],mlp_width=cfg['mlp_width'],
                          rtol=cfg['rtol'],atol=cfg['atol'],encoder_direction='backward',encoder_initialization='age',
                          init_width=cfg['init_width'],update_width=cfg['update_width'],decoder_width=cfg['decoder_width'],
                          encoder_ode_width=cfg['encoder_ode_width'],generator_ode_width=cfg['generator_ode_width'])
    if cfg['dataset_name']=='mimic':
        from mimic_experiment.baseline_models import ICUBaseline
        model=ICUBaseline(schema,kind,cfg['latent'],cfg['encoder_hidden'],cfg['ode_width'],cfg['rtol'],cfg['atol'])
    else:model=BaselineModel(schema,kind,cfg['latent'],cfg['encoder_hidden'],cfg['ode_width'],cfg['rtol'],cfg['atol'])
    # Mixed-field decoding supplies all task outputs. The old binary head was
    # frozen and unused; the new total contains only parameters serving this task.
    del model.core.classifier
    model.core.ode.net=field(cfg['latent'],cfg['generator_field_widths'])
    if kind=='latent_ode':model.core.encoder.ode.net=field(cfg['encoder_hidden'],cfg['encoder_field_widths'])
    return model
