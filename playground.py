import tensorflow as tf

# from matplotlib import pyplot as plt
#
# from dataset_utils import load_multi_domain_ds
#
# train_ds, test_ds = load_multi_domain_ds(
#     domains=["back", "left", "front", "right"],
#     should_augment_translation=False,
#     should_augment_hue=True)
#
#
# def show_examples(batch):
#     domain_images = batch
#     number_of_domains = len(domain_images)
#     number_of_examples = len(domain_images[0])
#
#     plt.figure(figsize=(2 * number_of_domains, 2 * number_of_examples))
#     for i in range(number_of_examples):
#         for j in range(number_of_domains):
#             index = i * number_of_domains + j + 1
#             domain_image = domain_images[j][i]
#             plt.subplot(number_of_examples, number_of_domains, index)
#             plt.imshow((domain_image + 1.) / 2.)
#             plt.axis("off")
#     plt.show()
#
#
# show_examples(next(iter(train_ds.take(1))))


a1 = tf.ragged.constant(
    [[[b'a1', b'a2', b'a3'], [b'b1', b'b2', b'b3'], [b'c1', b'c2', 'c3']], [[b'd1'], [b'e1', b'e2']]])
b1 = tf.ragged.constant([[[b't1', b't2', b't3'], [b'u1', b'u2'], [b'v1', b'v2']], [[b'w1'], [b'x1', b'x2', b'x3']]])

# @tf.function
# def tile_nd_ragged(a, b):
#     # Need a sentinel, otherwise it's hard to give it the initial shape we need.
#     # We'll drop the sentinel at the end.
#     acc = tf.ragged.constant([[[]]], dtype=a.dtype)
#
#     # Work one row at a time...
#     for i1 in range(len(a.nested_row_lengths()[0])):  # Should be able to write `for a1, b1 in zip(a, b)` soon.
#         a1 = a[i1]
#         b1 = b[i1]
#         # If the components have variable length, we can't use a TensorArray anymore,
#         # so use a RaggedTensor instead.
#         acc1 = tf.ragged.constant([[]], dtype=a.dtype)
#
#         # Do the actual tiling...
#         for i2 in tf.range(a1.nested_row_lengths()[0][i1]):
#             # Need this workaround to let tensors change shape in a loop.
#             tf.autograph.experimental.set_loop_options(
#                 shape_invariants=[(acc1, tf.TensorShape([None, None]))]
#             )
#
#             a2 = a1[i2]
#             b2 = b1[i2]
#             for _ in range(len(b2)):
#                 acc1 = tf.concat([acc1, tf.expand_dims(a2, 0)], axis=0)
#
#         acc1 = acc1[1:]  # Drop the sentinel.
#         acc = tf.concat([acc, tf.expand_dims(acc1, 0)], axis=0)  # Add the row to the final result.
#
#     acc = acc[1:]  # Drop the sentinel.
#     return acc
#
#
# print("a1", a1.shape)
# print("b1", b1.shape)
# result = tile_nd_ragged(a1, b1)
# print("result.shape", result.shape)
# print("result", result)

elems = tf.constant([[3], [5], [0], [2]])
result = tf.map_fn(tf.range, elems,
                   fn_output_signature=tf.RaggedTensorSpec(shape=[None],
                                                           dtype=tf.int32))

print("elems", elems)
print("result.shape", result.shape)
print("result", result)
