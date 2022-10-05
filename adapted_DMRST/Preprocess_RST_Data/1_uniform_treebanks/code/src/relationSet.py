#!/usr/bin/python
# -*- coding: utf-8 -*-


'''
Relation sets and mappings for the corpora annotated within the RST
'''


# General mapping for all corpora to the 18 coarse grained classes (Carlson et al. 2001)
mapping = {
    u'ahalbideratzea':'Enablement',
    u'alderantzizko-baldintza':'Condition',
    u'alternativa':'Condition',
    u'analogy':'Comparison',
    u'antitesia':'Contrast',
    u'antithesis':'Contrast',
    u'antítesis':'Contrast',
    u'arazo-soluzioa':'Topic-Comment',
    u'attribution':'Attribution',
    u'attribution-negative':'Attribution',
    u'aukera':'Condition',
    u'background':'Background',
    u'baldintza':'Condition',
    u'bateratzea':'Joint',
    u'birformulazioa':'Summary',
    u'capacitación':'Enablement',
    u'causa':'Cause',
    u'cause':'Cause',
    u'cause-result':'Cause',
    u'circumstance':'Background',
    u'circunstancia':'Background',
    u'comment':'Evaluation',
    u'comment-topic':'Topic-Comment',
    u'comparison':'Comparison',
    u'concesión':'Contrast',
    u'concession':'Contrast',
    u'conclusion':'Evaluation',
    u'condición':'Condition',
    u'condición-inversa':'Condition',
    u'condition':'Condition',
    u'conjunción':'Joint',
    u'conjunction':'Joint',
    u'consequence':'Cause',
    u'contingency':'Condition',
    u'contrast':'Contrast',
    u'contraste':'Contrast',
    u'definition':'Elaboration',
    u'definitu-gabeko-erlazioa':'Summary',
    u'disjunction':'Joint',
    u'disjuntzioa':'Joint',
    u'disyunción':'Joint',
    u'e-elaboration':'Elaboration',
    u'ebaluazioa':'Evaluation',
    u'ebidentzia':'Explanation',
    u'elaboración':'Elaboration',
    u'elaboration':'Elaboration',
    u'elaboration-additional':'Elaboration',
    u'elaboration-general-specific':'Elaboration',
    u'elaboration-object-attribute':'Elaboration',
    u'elaboration-part-whole':'Elaboration',
    u'elaboration-process-step':'Elaboration',
    u'elaboration-set-member':'Elaboration',
    u'elaborazioa':'Elaboration',
    u'enablement':'Enablement',
    u'evaluación':'Evaluation',
    u'evaluation':'Evaluation',
    u'evidence':'Explanation',
    u'evidencia':'Explanation',
    u'example':'Elaboration',
    u'explanation':'Explanation',
    u'explanation-argumentative':'Explanation',
    u'ez-baldintzatzailea':'Condition',
    u'fondo':'Background',
    u'helburua':'Enablement',
    u'hypothetical':'Condition',
    u'interpretación':'Evaluation',
    u'interpretation':'Evaluation',
    u'interpretazioa':'Evaluation',
    u'inverted-sequence':'Temporal',
    u'joint':'Joint',
    u'justificación':'Explanation',
    u'justifikazioa':'Explanation',
    u'justify':'Explanation',
    u'kausa':'Cause',
    u'konjuntzioa':'Joint',
    u'kontrastea':'Contrast',
    u'kontzesioa':'Contrast',
    u'laburpena':'Summary',
    u'list':'Joint',
    u'lista':'Joint',
    u'manner':'Manner-Means',
    u'means':'Manner-Means',
    u'medio':'Manner-Means',
    u'metodoa':'Manner-Means',
    u'motibazioa':'Explanation',
    u'motivación':'Explanation',
    u'motivation':'Explanation',
    u'non-volitional-cause':'Cause',
    u'non-volitional-result':'Cause',
    u'nonvolitional-cause':'Cause',
    u'nonvolitional-result':'Cause',
    u'ondorioa':'Cause',
    u'otherwise':'Condition',
    u'parenthetical':'Elaboration',
    u'preference':'Comparison',
    u'preparación':'Background',
    u'preparation':'Background',
    u'prestatzea':'Background',
    u'problem-solution':'Topic-Comment',
    u'proportion':'Comparison',
    u'propósito':'Enablement',
    u'purpose':'Enablement',
    u'question-answer':'Topic-Comment',
    u'reason':'Explanation',
    u'reformulación':'Summary',
    u'restatement':'Summary',
    u'restatement-mn':'Summary',
    u'result':'Cause',
    u'resultado':'Cause',
    u'resumen':'Summary',
    u'rhetorical-question':'Topic-Comment',
    u'same-unit':'Same-unit',
    u'secuencia':'Temporal',
    u'sekuentzia':'Temporal',
    u'sequence':'Temporal',
    u'solución':'Topic-Comment',
    u'solutionhood':'Topic-Comment',
    u'statement-response':'Topic-Comment',
    u'summary':'Summary',
    u'temporal-after':'Temporal',
    u'temporal-before':'Temporal',
    u'temporal-same-time':'Temporal',
    u'testuingurua':'Background',
    u'textual-organization':'Textual-Organization',
    u'textualorganization':'Textual-Organization',
    u'topic-comment':'Topic-Comment',
    u'topic-drift':'Topic-Change',
    u'topic-shift':'Topic-Change',
    u'unconditional':'Condition',
    u'unión':'Joint',
    u'unless':'Condition',
    u'volitional-cause':'Cause',
    u'volitional-result':'Cause',
    u'zirkunstantzia':'Background',
    u'question': 'Topic-Comment', # Shi Ke added for gum dataset
}


# English RST DT usual mapping
rstdt_classes = {
    'analogy':'comparison',
    'antithesis':'contrast',
    'attribution':'attribution',
    'attribution-negative':'attribution',
    'background':'background',
    'cause':'cause',
    'cause-result':'cause',
    'circumstance':'background',
    'comment':'evaluation',
    'comment-topic':'topic-comment',
    'comparison':'comparison',
    'concession':'contrast',
    'conclusion':'evaluation',
    'condition':'condition',
    'consequence':'cause',
    'contingency':'condition',
    'contrast':'contrast',
    'definition':'elaboration',
    'disjunction':'joint',
    'elaboration-additional':'elaboration',
    'elaboration-general-specific':'elaboration',
    'elaboration-object-attribute':'elaboration',
    'elaboration-part-whole':'elaboration',
    'elaboration-process-step':'elaboration',
    'elaboration-set-member':'elaboration',
    'enablement':'enablement',
    'evaluation':'evaluation',
    'evidence':'explanation',
    'example':'elaboration',
    'explanation-argumentative':'explanation',
    'hypothetical':'condition',
    'interpretation':'evaluation',
    'Interpretation':'evaluation',
    'inverted-sequence':'temporal',
    'list':'joint',
    'manner':'manner-means',
    'means':'manner-means',
    'otherwise':'condition',
    'preference':'comparison',
    'problem-solution':'topic-comment',
    'proportion':'comparison',
    'purpose':'enablement',
    'question-answer':'topic-comment',
    'reason':'explanation',
    'restatement':'summary',
    'result':'cause',
    'rhetorical-question':'topic-comment',
    'same-unit':'same-unit',
    'sequence':'temporal',
    'statement-response':'topic-comment',
    'summary':'summary',
    'temporal-after':'temporal',
    'temporal-before':'temporal',
    'temporal-same-time':'temporal',
    'textualorganization':'textual-organization',
    'topic-comment':'topic-comment',
    'topic-drift':'topic-change',
    'topic-shift':'topic-change'
}

rstdt2gum_classes = {'attribution': 'attribution', 'attribution-e': 'attribution', 'attribution-n': 'attribution', 'attribution-negative': 'attribution', 'attribution-positive': 'attribution', 'context': 'context', 'background': 'context', 'background-e': 'context', 'circumstance': 'context', 'circumstance-e': 'context', 'context-background': 'context', 'context-circumstance': 'context', 'causal': 'causal', 'cause': 'causal', 'cause-result': 'causal', 'result': 'causal', 'result-e': 'causal', 'consequence': 'causal', 'consequence-n-e': 'causal', 'consequence-n': 'causal', 'consequence-s-e': 'causal', 'consequence-s': 'causal', 'causal-cause': 'causal', 'causal-result': 'causal', 'contingency': 'contingency', 'condition': 'contingency', 'condition-e': 'contingency', 'hypothetical': 'contingency', 'otherwise': 'contingency', 'contingency-condition': 'contingency', 'adversative': 'adversative', 'contrast': 'adversative', 'concession': 'adversative', 'concession-e': 'adversative', 'antithesis': 'adversative', 'antithesis-e': 'adversative', 'adversative-antithesis': 'adversative', 'adversative-concession': 'adversative', 'adversative-contrast': 'adversative', 'elaboration': 'elaboration', 'elaboration-additional-e': 'elaboration', 'elaboration-general-specific': 'elaboration', 'elaboration-general-specific-e': 'elaboration', 'elaboration-part-whole': 'elaboration', 'elaboration-part-whole-e': 'elaboration', 'elaboration-process-step': 'elaboration', 'elaboration-process-step-e': 'elaboration', 'elaboration-object-attribute-e': 'elaboration', 'elaboration-object-attribute': 'elaboration', 'elaboration-set-member': 'elaboration', 'elaboration-set-member-e': 'elaboration', 'example': 'elaboration', 'example-e': 'elaboration', 'definition': 'elaboration', 'definition-e': 'elaboration', 'elaboration-attribute': 'elaboration', 'elaboration-additional': 'elaboration', 'evaluation': 'evaluation', 'evaluation-n': 'evaluation', 'evaluation-s-e': 'evaluation', 'evaluation-s': 'evaluation', 'interpretation': 'evaluation', 'interpretation-n': 'evaluation', 'interpretation-s-e': 'evaluation', 'interpretation-s': 'evaluation', 'conclusion': 'evaluation', 'comment': 'evaluation', 'comment-e': 'evaluation', 'evaluation-comment': 'evaluation', 'explanation': 'explanation', 'evidence': 'explanation', 'evidence-e': 'explanation', 'explanation-argumentative': 'explanation', 'explanation-argumentative-e': 'explanation', 'reason': 'explanation', 'reason-e': 'explanation', 'explanation-evidence': 'explanation', 'explanation-justify': 'explanation', 'explanation-motivation': 'explanation', 'joint': 'joint', 'list': 'joint', 'disjunction': 'joint', 'joint-disjunction': 'joint', 'joint-list': 'joint', 'joint-other': 'joint', 'joint-sequence': 'joint', 'topic-shift': 'joint', 'topic-drift': 'joint', 'temporal-before': 'joint', 'temporal-before-e': 'joint', 'temporal-after': 'joint', 'temporal-after-e': 'joint', 'temporal-same-time': 'joint', 'temporal-same-time-e': 'joint', 'sequence': 'joint', 'inverted-sequence': 'joint', 'comparison': 'joint', 'comparison-e': 'joint', 'preference': 'joint', 'preference-e': 'joint', 'analogy': 'joint', 'analogy-e': 'joint', 'proportion': 'joint', 'mode': 'mode', 'manner': 'mode', 'manner-e': 'mode', 'means': 'mode', 'means-e': 'mode', 'mode-manner': 'mode', 'mode-means': 'mode', 'organization': 'organization', 'organization-heading': 'organization', 'organization-phatic': 'organization', 'organization-preparation': 'organization', 'textualorganization': 'organization', 'purpose': 'purpose', 'purpose-attribute': 'purpose', 'purpose-goal': 'purpose', 'purpose-e': 'purpose', 'enablement': 'purpose', 'enablement-e': 'purpose', 'restatement': 'restatement', 'restatement-partial': 'restatement', 'restatement-repetition': 'restatement', 'summary': 'restatement', 'summary-n': 'restatement', 'summary-s': 'restatement', 'restatement-e': 'restatement', 'topic': 'topic', 'topic-question': 'topic', 'topic-solutionhood': 'topic', 'problem-solution': 'topic', 'problem-solution-n': 'topic', 'problem-solution-s': 'topic', 'question-answer': 'topic', 'question-answer-n': 'topic', 'question-answer-s': 'topic', 'statement-response': 'topic', 'statement-response-n': 'topic', 'statement-response-s': 'topic', 'topic-comment': 'topic', 'comment-topic': 'topic', 'rhetorical-question': 'topic', 'same-unit': 'same-unit'}

gum_classes = {
    "adversative-antithesis": "adversative",
    "adversative-concession": "adversative",
    "adversative-contrast": "adversative",
    "attribution-negative": "attribution",
    "attribution-positive": "attribution",
    "causal-cause": "causal",
    "causal-result": "causal",
    "context-background": "context",
    "context-circumstance": "context",
    "contingency-condition": "contingency",
    "elaboration-additional": "elaboration",
    "elaboration-attribute": "elaboration",
    "evaluation-comment": "evaluation",
    "explanation-evidence": "explanation",
    "explanation-justify": "explanation",
    "explanation-motivation": "explanation",
    "joint-disjunction": "joint",
    "joint-list": "joint",
    "joint-other": "joint",
    "joint-sequence": "joint",
    "mode-manner": "mode",
    "mode-means": "mode",
    "organization-heading": "organization",
    "organization-phatic": "organization",
    "organization-preparation": "organization",
    "purpose-attribute": "purpose",
    "purpose-goal": "purpose",
    "restatement-partial": "restatement",
    "restatement-repetition": "restatement",
    "same-unit": "same-unit",
    "topic-question": "topic",
    "topic-solutionhood": "topic",
}


gum2rstdt_classes = {
    "adversative-antithesis": "contrast",
    "adversative-concession": "contrast",
    "adversative-contrast": "contrast",
    "attribution-negative": "attribution",
    "attribution-positive": "attribution",
    "causal-cause": "cause",
    "causal-result": "cause",
    "context-background": "background",
    "context-circumstance": "background",
    "contingency-condition": "condition",
    "elaboration-additional": "elaboration",
    "elaboration-attribute": "elaboration",
    "evaluation-comment": "evaluation",
    "explanation-evidence": "explanation",
    "explanation-justify": "explanation",
    "explanation-motivation": "explanation",
    "joint-disjunction": "joint",
    "joint-list": "joint",
    "joint-other": "topic-change",
    "joint-sequence": "temporal",
    "mode-manner": "manner-means",
    "mode-means": "manner-means",
    "organization-heading": "textual-organization",
    "organization-phatic": "topic-comment",
    "organization-preparation": "textual-organization",
    "purpose-attribute": "elaboration",
    "purpose-goal": "enablement",
    "restatement-partial": "summary",
    "restatement-repetition": "summary",
    "same-unit": "same-unit",
    "topic-question": "topic-comment",
    "topic-solutionhood": "topic-comment",
}


# Dictionnaries with original labels as annotated in the data and possible
# corrections if needed (ie translation to English, correction of errors)

basque_labels = {
    u'ahalbideratzea':u'enablement',
    u'alderantzizko-baldintza':u'unless',
    u'antitesia':u'antithesis',
    u'arazo-soluzioa':u'solutionhood',#solution-hood changed to 'solutionhood'
    u'aukera':u'otherwise',
    u'baldintza':u'condition',
    u'bateratzea':u'joint',#Join on the website but Joint is a most common name
    u'birformulazioa':u'restatement',
    u'definitu-gabeko-erlazioa':u'reformulation',
    u'disjuntzioa':u'disjunction',
    u'ebaluazioa':u'evaluation',
    u'ebidentzia':u'evidence',
    u'elaborazioa':u'elaboration',
    u'ez-baldintzatzailea':u'unconditional',
    u'helburua':u'purpose',
    u'interpretazioa':u'interpretation',
    u'justifikazioa':u'justify',
    u'kausa':u'cause',
    u'konjuntzioa':u'conjunction',
    u'kontrastea':u'contrast',
    u'kontzesioa':u'concession',
    u'laburpena':u'summary',
    u'lista':u'list',
    u'metodoa':u'means',
    u'motibazioa':u'motivation',
    u'ondorioa':u'result',
    u'prestatzea':u'preparation',
    u'same-unit':u'same-unit',
    u'sekuentzia':u'sequence',
    u'testuingurua':u'background',
    u'zirkunstantzia':u'circumstance'
}

brazilianCst_labels = {#The most similar to the RST
    u'antithesis':u'antithesis',
    u'attribution':u'attribution',
    u'background':u'background',
    u'circumstance':u'circumstance',
    u'comparison':u'comparison',
    u'concession':u'concession',
    u'conclusion':u'conclusion',
    u'condition':u'condition',
    u'contrast':u'contrast',
    u'elaboration':u'elaboration',
    u'enablement':u'enablement',
    u'evaluation':u'evaluation',
    u'evidence':u'evidence',
    u'explanation':u'explanation',
    u'interpretation':u'interpretation',
    u'joint':u'joint',
    u'justify':u'justify',
    u'list':u'list',
    u'means':u'means',
    u'motivation':u'motivation',
    u'non-volitional-cause':u'non-volitional-cause',
    u'non-volitional-result':u'non-volitional-result',
    u'otherwise':u'otherwise',
    u'parenthetical':u'parenthetical',
    u'purpose':u'purpose',
    u'restatement':u'restatement',
    u'same-unit':u'same-unit',
    u'sequence':u'sequence',
    u'solutionhood':u'solutionhood',
    u'volitional-cause':u'volitional-cause',
    u'volitional-result':u'volitional-result'
}

brazilianSum_labels = {
    u'antithesis':u'antithesis',
    u'attribution':u'attribution',
    u'background':u'background',
    u'circumstance':u'circumstance',
    u'comparison':u'comparison',
    u'concession':u'concession',
    u'conclusion':u'conclusion',
    u'condition':u'condition',
    u'contrast':u'contrast',
    u'elaboration':u'elaboration',
    u'evaluation':u'evaluation',
    u'evidence':u'evidence',
    u'explanation':u'explanation',
    u'interpretation':u'interpretation',
    u'joint':u'joint',
    u'justify':u'justify',
    u'list':u'list',
    u'means':u'means',
    u'non-volitional-cause':u'non-volitional-cause',
    u'non-volitional-result':u'non-volitional-result',
    u'otherwise':u'otherwise',
    u'parenthetical':u'parenthetical',
    u'purpose':u'purpose',
    u'restatement':u'restatement',
    u'same-unit':u'same-unit',
    u'sequence':u'sequence',
    u'solutionhood':u'solutionhood',
    u'summary':u'summary',
    u'volitional-cause':u'volitional-cause'
}

brazilianTCC_labels = {
    u'antithesis':u'antithesis',
    u'attribution':u'attribution',
    u'background':u'background',
    u'circumstance':u'circumstance',
    u'comparison':u'comparison',
    u'concession':u'concession',
    u'conclusion':u'conclusion',
    u'condition':u'condition',
    u'contrast':u'contrast',
    u'elaboration':u'elaboration',
    u'enablement':u'enablement',
    u'evaluation':u'evaluation',
    u'evidence':u'evidence',
    u'explanation':u'explanation',
    u'interpretation':u'interpretation',
    u'joint':u'joint',
    u'justify':u'justify',
    u'list':u'list',
    u'means':u'means',
    u'motivation':u'motivation',
    u'non-volitional-cause':u'non-volitional-cause',
    u'non-volitional-result':u'non-volitional-result',
    u'otherwise':u'otherwise',
    u'parenthetical':u'parenthetical',
    u'purpose':u'purpose',
    u'restatement':u'restatement',
    u'same-unit':u'same-unit',
    u'sequence':u'sequence',
    u'solutionhood':u'solutionhood',
    u'summary':u'summary',
    u'volitional-cause':u'volitional-cause',
    u'volitional-result':u'volitional-result'
}


germanPcc_labels = {
    u'antithesis':u'antithesis',
    u'background':u'background',
    u'cause':u'cause',
    u'circumstance':u'circumstance',
    u'concession':u'concession',
    u'condition':u'condition',
    u'conjunction':u'conjunction',
    u'contrast':u'contrast',
    u'disjunction':u'disjunction',
    u'e-elaboration':u'entity-elaboration',
    u'elaboration':u'elaboration',
    u'enablement':u'enablement',
    u'evaluation':u'evaluation',
    u'evidence':u'evidence',
    u'interpretation':u'interpretation',
    u'joint':u'joint',
    u'justify':u'justify',
    u'list':u'list',
    u'means':u'means',
    u'motivation':u'motivation',
    u'otherwise':u'otherwise',
    u'preparation':u'preparation',
    u'purpose':u'purpose',
    u'reason':u'reason',
    u'restatement':u'restatement',
    u'result':u'result',
    u'sequence':u'sequence',
    u'solutionhood':u'solutionhood',
    u'summary':u'summary',
    u'unless':u'unless'
}

spanish_labels = {
    u'alternativa':u'alternative',
    u'antítesis':u'antithesis',
    u'capacitación':u'enablement',
    u'causa':u'cause',
    u'circunstancia':u'circumstance',
    u'concesión':u'concession',
    u'condición':u'condition',
    u'condición-inversa':u'unconditional',
    u'conjunción':u'conjunction',
    u'contraste':u'contrast',
    u'disyunción':u'disjunction',
    u'elaboración':u'elaboration',
    u'evaluación':u'evaluation',
    u'evidencia':u'evidence',
    u'fondo':u'background',
    u'interpretación':u'interpretation',
    u'justificación':u'justify',
    u'lista':u'list',
    u'medio':u'means',
    u'motivación':u'motivation',
    u'preparación':u'preparation',
    u'propósito':u'purpose',
    u'reformulación':u'restatement',
    u'resultado':u'result',
    u'resumen':u'summary',
    u'same-unit':u'same-unit',
    u'secuencia':u'sequence',
    u'solución':u'solutionhood',
    u'unión':u'union',
    u'unless':u'unless'
}

dutch_labels = {
    u'antithesis':u'antithesis',
    u'background':u'background',
    u'circumstance':u'circumstance',
    u'concession':u'concession',
    u'condition':u'condition',
    u'conjunction':u'conjunction',
    u'contrast':u'contrast',
    u'disjunction':u'disjunction',
    u'elaboration':u'elaboration',
    u'enablement':u'enablement',
    u'evaluation':u'evaluation',
    u'evidence':u'evidence',
    u'interpretation':u'interpretation',
    u'joint':u'joint',
    u'justify':u'justify',
    u'list':u'list',
    u'means':u'means',
    u'motivation':u'motivation',
    u'nonvolitional-cause':u'non-volitional-cause',
    u'nonvolitional-result':u'non-volitional-result',
    u'otherwise':u'otherwise',
    u'preparation':u'preparation',
    u'purpose':u'purpose',
    u'restatement':u'restatement',
    u'restatement-mn':u'restatement',# multinuclear version
    u'sequence':u'sequence',
    u'solutionhood':u'solutionhood',
    u'summary':u'summary',
    u'unconditional':u'unconditional',
    u'unless':u'unless',
    u'volitional-cause':u'volitional-cause',
    u'volitional-result':u'volitional-result'
}


