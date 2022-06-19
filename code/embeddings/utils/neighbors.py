from . import ast_utils as utils


MAX_SIZE = 6
MAX_DEPTH = 3


def with_depth(func):
    ''' A decorator used to automatically maintain node depth when applying 
        recursive func function to a tree. '''
    def recursive_with_depth(node, *args, depth=0, **kwargs):
        if node.label() != 'list':
            depth += 1
        yield from func(node, *args, depth=depth, **kwargs)
    return recursive_with_depth


def generate_recursively(func=None, depth=False):
    if func is None:
        return lambda f : generate_recursively(f, depth)
    ''' A decorator used to apply func recursively to every child of a tree. '''
    def recursive(node, *args, **kwargs):

        continue_recursion = yield from (
            r for r in func(node, *args, **kwargs) if r is not None
        )

        if continue_recursion is False:
            return

        yield from (
            node.with_ith_child(i, new_child)
            for i, child in enumerate(node.children())
            for new_child in recursive(child, *args, **kwargs)
        )

    if depth:
        recursive = with_depth(recursive)
    return recursive


@generate_recursively
def with_changed_parameter(node: utils.ASTNode):
    ''' Generates all possible ASTs obtainable by changing parameter (if any) of
        any node in the tree. '''
    if node.label_enum() == utils.NodeType.LIST:
        return
    node_type, node_parameter = utils.split_node_type_parameter(node.label())
    new_labels = (
        utils.merge_node_type_parameter(node_type, new_parameter)
        for new_parameter in utils.parameters_for_node_type(node_type)
        if new_parameter != node_parameter
    )
    yield from (node.with_label(label) for label in new_labels)


@generate_recursively
def with_inserted_node(node: utils.ASTNode, dataset_id: str):
    ''' Generates all possible ASTs obtainable by inserting a new single node
        as a child anywhere in the tree. '''
    if node.label_enum() != utils.NodeType.LIST:
        return


    new_nodes = (
        utils.create_empty_node(node_type, parameter)
        for node_type in utils.node_types_for_dataset(dataset_id)
        for parameter in utils.node_type_parameters.get(node_type, [None])
    )
        
        
    yield from (
        node.with_inserted_child(i, new_node)
        for new_node in new_nodes
        for i in range(0, node.n_children()+1)
    )


def generate_all_detachments(node: utils.ASTNode):
    ''' Generate all possible ways to detach a tail of some child list of some
        node in the tree. '''
    def detach_tail(children, i):
        head_children = children[:i]
        head_node = node.with_children(head_children)
        # Detach the tail of the list:
        detached = children[i:]
        return (head_node, head_node, detached)

    if node.label_enum() == utils.NodeType.LIST:
        yield from (
            detach_tail(node.children(), i)
            for i in range(0, node.n_children())
        )
 
    yield from(
        (node.with_ith_child(i, new_child), detachment_point, detached)
        for i, child in enumerate(node.children())
        for new_child, detachment_point, detached in generate_all_detachments(child)
    )


@generate_recursively(depth=True)
def with_moved_tail(
    node: utils.ASTNode,
    detachment_point: utils.ASTNode,
    detached,
    detached_depth,
    max_depth,
    depth,
):
    ''' Generate all possible ways to insert the detached tail somewhere in the
        tree, avoiding insertion to the same place from where it was detached.
    '''
    if node.label_enum() != utils.NodeType.LIST:
        return

    if depth + detached_depth > max_depth:
        return False

    allow_last = (id(node) != id(detachment_point))

    yield from (
        node.with_inserted_children(i, detached)
        for i in range(0, node.n_children() + int(allow_last))
    )
        

def with_deleted_or_moved_tail(node: utils.ASTNode, max_depth):
    ''' Generate all possible ways to detach some tail from the tree and then
        either remove it or attach it to some other place. '''
    def get_detached_depth(detached):
        detached_depth = 0
        for detached_node in detached:
            detached_depth = max(detached_depth, detached_node.depth())
        return detached_depth

    def deleted_and_moved(root, detachment_point, detached):
        yield root
        yield from with_moved_tail(root, detachment_point, detached, get_detached_depth(detached), max_depth)

    yield from (
        neighbor
        for root, detachment_point, detached in generate_all_detachments(node)
        for neighbor in deleted_and_moved(root, detachment_point, detached)
    )


@generate_recursively
def with_deleted_node(node: utils.ASTNode):
    if node.label_enum() != utils.NodeType.LIST:
        return

    yield from (
        node.with_removed_child(i)
        for i in range(0, node.n_children())
    )


@generate_recursively(depth=True)
def with_enclosing_forever(node: utils.ASTNode, max_depth, depth):
    if node.label_enum() != utils.NodeType.LIST:
        return

    if node.n_children() == 0:
        return

    last_child = node.children()[-1]
    if depth + last_child.depth() > max_depth:
        return False

    enclosed_in_forever = utils.ASTNode('maze_forever', [
        utils.ASTNode('list', [last_child]),
    ])

    yield node.with_ith_child(-1, enclosed_in_forever)


@generate_recursively
def without_forever(node: utils.ASTNode):
    if node.label_enum() != utils.NodeType.FOREVER:
        return

    child_list = node.children()[0]
    if child_list.n_children() != 1:
        return

    child = child_list.children()[0]
    yield child


def check_size_and_depth(
    node: utils.ASTNode,
    max_size=MAX_SIZE,
    max_depth=MAX_DEPTH,
):
    return node.size() <= max_size and node.depth() <= max_depth


def generate_neighbors(
    node: utils.ASTNode,
    dataset_id: str,
    max_size=MAX_SIZE,
    max_depth=MAX_DEPTH,
    max_n_if_else_and_forever=None,
):
    if not check_size_and_depth(node, max_size, max_depth):
        return

    yield from with_changed_parameter(node)

    yield from (
        neighbor for neighbor in with_inserted_node(node, dataset_id)
        if check_size_and_depth(neighbor, max_size, max_depth)
    )

    #yield from with_deleted_or_moved_tail(node, max_depth)
    yield from with_deleted_node(node)
    if dataset_id.endswith('18'):
        if node.size() < max_size and max_n_if_else_and_forever is not None and node._n_if_else + node._n_forever < max_n_if_else_and_forever:
            yield from with_enclosing_forever(node, max_depth)
        if node._n_forever > 0:
            yield from without_forever(node)


def constrained_generate_neighbors(
    node: utils.ASTNode,
    dataset_id: str,
    constraints=None,
    max_size=MAX_SIZE,
    max_depth=MAX_DEPTH,
):
    if constraints is None or not hasattr(constraints, 'max_n_if_else_and_forever'):
        max_n_if_else_and_forever = None
    else:
        max_n_if_else_and_forever = constraints.max_n_if_else_and_forever

    if constraints is None:
        yield from generate_neighbors(node, dataset_id, max_size, max_depth, max_n_if_else_and_forever)
    else:
        yield from (
            neighbor for neighbor in generate_neighbors(node, dataset_id, max_size, max_depth, max_n_if_else_and_forever)
            if constraints(neighbor)
        )


def generate_unique_neighbors(
    node: utils.ASTNode,
    dataset_id: str,
    max_size=MAX_SIZE,
    max_depth=MAX_DEPTH,
):
    already_generated = set()
    for neighbor in generate_neighbors(node, dataset_id, max_size, max_depth):
        if neighbor not in already_generated:
            already_generated.add(neighbor)
            yield neighbor


def chain_neighborhood(
    node,
    dataset_id,
    distance,
    max_size=MAX_SIZE,
    max_depth=MAX_DEPTH,
):
    ''' Generates all neighbors at the provided distance. '''
    if distance == 0:
        yield node
        return

    for neighbor in generate_neighbors(node, dataset_id, max_size, max_depth):
        yield from chain_neighborhood(
            neighbor, dataset_id, distance-1, max_size, max_depth)

