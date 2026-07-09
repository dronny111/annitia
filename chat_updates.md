Why is age allowed to be used as features? Particularly for death event you can simply rank the censored patients simply based on the last visited date and you get the perfect Concordance Index score for this data subset. Around 70% of the patient belong to censored group  in death category. 

Author Balram S |Created at 4/14/2026, 4:20:26 AM
This is not data leakage in the strict sense - the data is legitimately available at prediction time.
 

Our goal is not early prediction but risk assessment at various stages of a patient's care journey, where all accumulated visit history is naturally available. Using longitudinal features is therefore entirely valid in this context.

However, designing features that disentangle this thing is precisely the kind of smart approach that will be rewarded in the subjective scoring component of the challenge.