
# _parsetab.py
# This file is automatically generated. Do not edit.
# pylint: disable=W,C,R
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = 'C_LBRACE C_RBRACE DEF ELSE E_LBRACE E_RBRACE FRONT_IS_CLEAR IF IFELSE INT I_LBRACE I_RBRACE LEFT_IS_CLEAR MARKERS_PRESENT MOVE M_LBRACE M_RBRACE NO_FRONT_IS_CLEAR NO_LEFT_IS_CLEAR NO_MARKERS_PRESENT NO_RIGHT_IS_CLEAR PICK_MARKER PUT_MARKER REPEAT RIGHT_IS_CLEAR RUN R_LBRACE R_RBRACE TURN_LEFT TURN_RIGHT WHILE W_LBRACE W_RBRACEprog : DEF RUN M_LBRACE stmt M_RBRACEstmt : while\n                | repeat\n                | stmt_stmt\n                | action\n                | if\n                | ifelse\n        stmt_stmt : stmt stmt\n        if : IF C_LBRACE cond C_RBRACE I_LBRACE stmt I_RBRACE\n        ifelse : IFELSE C_LBRACE cond C_RBRACE I_LBRACE stmt I_RBRACE ELSE E_LBRACE stmt E_RBRACE\n        while : WHILE C_LBRACE cond C_RBRACE W_LBRACE stmt W_RBRACE\n        repeat : REPEAT cste R_LBRACE stmt R_RBRACE\n        cond : FRONT_IS_CLEAR\n                | NO_FRONT_IS_CLEAR\n                | LEFT_IS_CLEAR\n                | NO_LEFT_IS_CLEAR\n                | RIGHT_IS_CLEAR\n                | NO_RIGHT_IS_CLEAR\n                | MARKERS_PRESENT\n                | NO_MARKERS_PRESENT\n        action : MOVE\n                  | TURN_RIGHT\n                  | TURN_LEFT\n                  | PICK_MARKER\n                  | PUT_MARKER\n        cste : INT\n        '
    
_lr_action_items = {'DEF':([0,],[2,]),'$end':([1,22,],[0,-1,]),'RUN':([2,],[3,]),'M_LBRACE':([3,],[4,]),'WHILE':([4,5,6,7,8,9,10,11,14,15,16,17,18,21,37,41,44,45,46,47,48,49,50,51,52,55,56,57,],[12,12,-2,-3,-4,-5,-6,-7,-21,-22,-23,-24,-25,12,12,12,12,-12,12,12,12,12,12,-11,-9,12,12,-10,]),'REPEAT':([4,5,6,7,8,9,10,11,14,15,16,17,18,21,37,41,44,45,46,47,48,49,50,51,52,55,56,57,],[13,13,-2,-3,-4,-5,-6,-7,-21,-22,-23,-24,-25,13,13,13,13,-12,13,13,13,13,13,-11,-9,13,13,-10,]),'MOVE':([4,5,6,7,8,9,10,11,14,15,16,17,18,21,37,41,44,45,46,47,48,49,50,51,52,55,56,57,],[14,14,-2,-3,-4,-5,-6,-7,-21,-22,-23,-24,-25,14,14,14,14,-12,14,14,14,14,14,-11,-9,14,14,-10,]),'TURN_RIGHT':([4,5,6,7,8,9,10,11,14,15,16,17,18,21,37,41,44,45,46,47,48,49,50,51,52,55,56,57,],[15,15,-2,-3,-4,-5,-6,-7,-21,-22,-23,-24,-25,15,15,15,15,-12,15,15,15,15,15,-11,-9,15,15,-10,]),'TURN_LEFT':([4,5,6,7,8,9,10,11,14,15,16,17,18,21,37,41,44,45,46,47,48,49,50,51,52,55,56,57,],[16,16,-2,-3,-4,-5,-6,-7,-21,-22,-23,-24,-25,16,16,16,16,-12,16,16,16,16,16,-11,-9,16,16,-10,]),'PICK_MARKER':([4,5,6,7,8,9,10,11,14,15,16,17,18,21,37,41,44,45,46,47,48,49,50,51,52,55,56,57,],[17,17,-2,-3,-4,-5,-6,-7,-21,-22,-23,-24,-25,17,17,17,17,-12,17,17,17,17,17,-11,-9,17,17,-10,]),'PUT_MARKER':([4,5,6,7,8,9,10,11,14,15,16,17,18,21,37,41,44,45,46,47,48,49,50,51,52,55,56,57,],[18,18,-2,-3,-4,-5,-6,-7,-21,-22,-23,-24,-25,18,18,18,18,-12,18,18,18,18,18,-11,-9,18,18,-10,]),'IF':([4,5,6,7,8,9,10,11,14,15,16,17,18,21,37,41,44,45,46,47,48,49,50,51,52,55,56,57,],[19,19,-2,-3,-4,-5,-6,-7,-21,-22,-23,-24,-25,19,19,19,19,-12,19,19,19,19,19,-11,-9,19,19,-10,]),'IFELSE':([4,5,6,7,8,9,10,11,14,15,16,17,18,21,37,41,44,45,46,47,48,49,50,51,52,55,56,57,],[20,20,-2,-3,-4,-5,-6,-7,-21,-22,-23,-24,-25,20,20,20,20,-12,20,20,20,20,20,-11,-9,20,20,-10,]),'M_RBRACE':([5,6,7,8,9,10,11,14,15,16,17,18,21,45,51,52,57,],[22,-2,-3,-4,-5,-6,-7,-21,-22,-23,-24,-25,-8,-12,-11,-9,-10,]),'R_RBRACE':([6,7,8,9,10,11,14,15,16,17,18,21,41,45,51,52,57,],[-2,-3,-4,-5,-6,-7,-21,-22,-23,-24,-25,-8,45,-12,-11,-9,-10,]),'W_RBRACE':([6,7,8,9,10,11,14,15,16,17,18,21,45,48,51,52,57,],[-2,-3,-4,-5,-6,-7,-21,-22,-23,-24,-25,-8,-12,51,-11,-9,-10,]),'I_RBRACE':([6,7,8,9,10,11,14,15,16,17,18,21,45,49,50,51,52,57,],[-2,-3,-4,-5,-6,-7,-21,-22,-23,-24,-25,-8,-12,52,53,-11,-9,-10,]),'E_RBRACE':([6,7,8,9,10,11,14,15,16,17,18,21,45,51,52,56,57,],[-2,-3,-4,-5,-6,-7,-21,-22,-23,-24,-25,-8,-12,-11,-9,57,-10,]),'C_LBRACE':([12,19,20,],[23,26,27,]),'INT':([13,],[25,]),'FRONT_IS_CLEAR':([23,26,27,],[29,29,29,]),'NO_FRONT_IS_CLEAR':([23,26,27,],[30,30,30,]),'LEFT_IS_CLEAR':([23,26,27,],[31,31,31,]),'NO_LEFT_IS_CLEAR':([23,26,27,],[32,32,32,]),'RIGHT_IS_CLEAR':([23,26,27,],[33,33,33,]),'NO_RIGHT_IS_CLEAR':([23,26,27,],[34,34,34,]),'MARKERS_PRESENT':([23,26,27,],[35,35,35,]),'NO_MARKERS_PRESENT':([23,26,27,],[36,36,36,]),'R_LBRACE':([24,25,],[37,-26,]),'C_RBRACE':([28,29,30,31,32,33,34,35,36,38,39,],[40,-13,-14,-15,-16,-17,-18,-19,-20,42,43,]),'W_LBRACE':([40,],[44,]),'I_LBRACE':([42,43,],[46,47,]),'ELSE':([53,],[54,]),'E_LBRACE':([54,],[55,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'prog':([0,],[1,]),'stmt':([4,5,21,37,41,44,46,47,48,49,50,55,56,],[5,21,21,41,21,48,49,50,21,21,21,56,21,]),'while':([4,5,21,37,41,44,46,47,48,49,50,55,56,],[6,6,6,6,6,6,6,6,6,6,6,6,6,]),'repeat':([4,5,21,37,41,44,46,47,48,49,50,55,56,],[7,7,7,7,7,7,7,7,7,7,7,7,7,]),'stmt_stmt':([4,5,21,37,41,44,46,47,48,49,50,55,56,],[8,8,8,8,8,8,8,8,8,8,8,8,8,]),'action':([4,5,21,37,41,44,46,47,48,49,50,55,56,],[9,9,9,9,9,9,9,9,9,9,9,9,9,]),'if':([4,5,21,37,41,44,46,47,48,49,50,55,56,],[10,10,10,10,10,10,10,10,10,10,10,10,10,]),'ifelse':([4,5,21,37,41,44,46,47,48,49,50,55,56,],[11,11,11,11,11,11,11,11,11,11,11,11,11,]),'cste':([13,],[24,]),'cond':([23,26,27,],[28,38,39,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> prog","S'",1,None,None,None),
  ('prog -> DEF RUN M_LBRACE stmt M_RBRACE','prog',5,'p_prog','parser_karel_unified.py',116),
  ('stmt -> while','stmt',1,'p_stmt','parser_karel_unified.py',125),
  ('stmt -> repeat','stmt',1,'p_stmt','parser_karel_unified.py',126),
  ('stmt -> stmt_stmt','stmt',1,'p_stmt','parser_karel_unified.py',127),
  ('stmt -> action','stmt',1,'p_stmt','parser_karel_unified.py',128),
  ('stmt -> if','stmt',1,'p_stmt','parser_karel_unified.py',129),
  ('stmt -> ifelse','stmt',1,'p_stmt','parser_karel_unified.py',130),
  ('stmt_stmt -> stmt stmt','stmt_stmt',2,'p_stmt_stmt','parser_karel_unified.py',140),
  ('if -> IF C_LBRACE cond C_RBRACE I_LBRACE stmt I_RBRACE','if',7,'p_if','parser_karel_unified.py',150),
  ('ifelse -> IFELSE C_LBRACE cond C_RBRACE I_LBRACE stmt I_RBRACE ELSE E_LBRACE stmt E_RBRACE','ifelse',11,'p_ifelse','parser_karel_unified.py',185),
  ('while -> WHILE C_LBRACE cond C_RBRACE W_LBRACE stmt W_RBRACE','while',7,'p_while','parser_karel_unified.py',220),
  ('repeat -> REPEAT cste R_LBRACE stmt R_RBRACE','repeat',5,'p_repeat','parser_karel_unified.py',246),
  ('cond -> FRONT_IS_CLEAR','cond',1,'p_cond','parser_karel_unified.py',258),
  ('cond -> NO_FRONT_IS_CLEAR','cond',1,'p_cond','parser_karel_unified.py',259),
  ('cond -> LEFT_IS_CLEAR','cond',1,'p_cond','parser_karel_unified.py',260),
  ('cond -> NO_LEFT_IS_CLEAR','cond',1,'p_cond','parser_karel_unified.py',261),
  ('cond -> RIGHT_IS_CLEAR','cond',1,'p_cond','parser_karel_unified.py',262),
  ('cond -> NO_RIGHT_IS_CLEAR','cond',1,'p_cond','parser_karel_unified.py',263),
  ('cond -> MARKERS_PRESENT','cond',1,'p_cond','parser_karel_unified.py',264),
  ('cond -> NO_MARKERS_PRESENT','cond',1,'p_cond','parser_karel_unified.py',265),
  ('action -> MOVE','action',1,'p_action','parser_karel_unified.py',275),
  ('action -> TURN_RIGHT','action',1,'p_action','parser_karel_unified.py',276),
  ('action -> TURN_LEFT','action',1,'p_action','parser_karel_unified.py',277),
  ('action -> PICK_MARKER','action',1,'p_action','parser_karel_unified.py',278),
  ('action -> PUT_MARKER','action',1,'p_action','parser_karel_unified.py',279),
  ('cste -> INT','cste',1,'p_cste','parser_karel_unified.py',288),
]