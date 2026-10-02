"""Training-only source themes, independent of the development error list."""
TOPICS = {
 'daily': ['a tenant arranging access to a storage cupboard', 'a neighbor returning borrowed gardening equipment',
  'a coworker arranging a change to a meeting agenda', 'a household sorting clothes for donation',
  'a customer explaining a printing request', 'a volunteer allocating tasks for a community event',
  'a friend arranging collection of a repaired bicycle', 'a family comparing two ways to organize a shared calendar'],
 'travel': ['a visitor asking where to leave a stroller during a tour', 'a passenger checking the luggage allowance for an excursion',
  'a traveler asking about luggage delivery to accommodation', 'a guest arranging an early breakfast before leaving',
  'a cyclist asking about a route with steep inclines', 'a group agreeing where to meet after visiting a market',
  'a passenger explaining why a travel document is in another bag', 'a tourist asking how to reach an observation deck'],
 'food': ['preparing oats with fruit kept separate until serving', 'using a spatula to turn delicate fish in a pan',
  'kneading dough and leaving it to rest before shaping', 'a customer distinguishing a soup portion from a tasting portion',
  'preparing a sauce by stirring flour into melted fat', 'a cook cooling rice before adding it to a salad',
  'a diner requesting a separate container for leftover noodles', 'a baker wrapping cooled biscuits for transport'],
 'academic': ['a statistics class comparing population and sample descriptions', 'a physics class distinguishing distance from displacement',
  'a chemistry class describing a reversible change of phase', 'a biology class discussing the function of a membrane',
  'a geography class comparing erosion and deposition', 'a computing class explaining a queue versus a stack',
  'a history class distinguishing a quotation from a later interpretation', 'a mathematics class describing a necessary versus a sufficient condition'],
 'hard': ['culture: a laboratory discussing growth in a culture vessel', 'culture: a sociologist discussing customs in an organization',
  'interest: a savings account accumulating interest', 'interest: a reader developing interest in a new subject',
  'field: a database administrator inspecting a record field', 'field: a farmer discussing a recently planted field',
  'induction: a manager arranging an employee induction session', 'induction: a mathematics student reviewing a proof by induction'],
}
HARD_GROUPS = ['v5-seed-20','v5-seed-20','v5-seed-21','v5-seed-21','v5-seed-23','v5-seed-23','v5-seed-24','v5-seed-24']
