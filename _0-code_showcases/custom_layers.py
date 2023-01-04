#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 12:10:09 2021

@author: marlinberger

script contains custom layers for neural networks
"""

# python packages
import logging
import numpy as np
from tensorflow import keras
import tensorflow as tf

# modules from this package
from . import coordinate_position_assignment as cpa


def check_import():
    """use this function to check if the import worked
    """
    logging.debug("\nyou made it in\n")


# decorator ensures easy loading of the custom layers
@keras.utils.register_keras_serializable(package="Custom", name="NaNs_to_zero")
class NaN_to_zero(keras.layers.Layer):
    """layer will convert all NaN's in a given instance/batch to 0's
    """

    def __init__(self, name="NaNs_to_zero", **kwargs):
        """initialise everything needed for this layer, which is basicly nothing
        """
        super().__init__(name=name, **kwargs)

    def call(self, X):
        """heart of the layer: every batch will run through the graph that is
        created here

        Args:
            X (tf.Tensor): a given batch.
                           shape: [batch_size, seq_len, frame_repr_len]
        Returns:
            X (tf.Tensor): the batch with nan's converted to 0
                           shape: [batch_size, seq_len, frame_repr_len]
        """
        # mask nan values
        value_not_nan = tf.dtypes.cast(
            tf.math.logical_not(
                tf.math.is_nan(X)
            ),
            dtype=tf.float32
        )
        # this tf method return 0 for X if value_not_nan is 0, no matter what
        # X is originally. As this also works if X is nan, this method works
        X = tf.math.multiply_no_nan(X, value_not_nan)
        return (X)

    def compute_output_shape(self, input_shape):
        """this function is needed to seemingly use the custom layer

        Args:
            input_shape
        Returns:
            input_shape
        """
        return (input_shape)

    def get_config(self):
        """this function is needed to seemingly use the custom layer

        Returns:
            _configs (dict): an assignment of all variables that were used to
                             initialise the layer
        """
        config = {
            # nothing (yet)
        }
        base_config = super().get_config()
        return (dict(list(base_config.items()) + list(config.items())))


# decorator ensures easy loading of the custom layers
@keras.utils.register_keras_serializable(package="Custom", name="Normalizer")
class Normalizer(keras.layers.Layer):
    """provide different normalization strategies, which will not only ensure
    normalized data, but also how general a net, trained with a choosen
    strategy, can opereate.
    The strategies provided are listed from very setup&case specific to most
    general. Note that for a general approach, the dataset used for training
    should use a bunch of randomized data augmentation technics.

    Strategy 1 (pic_bound):
        Normalize correlated to the x-y space of the given of the picture.

    Strategy 2 (seq_on_last):
        Normalize coordinates (seperated per hand) to the wrist points of the
        most current point which is available for this hand

    Strategy 3 (inst_on_self):
        Normalize all coordinates related to themselves, sothat they appear
        stational.
    """

    def __init__(self, strategy="pic_bound", csv_struct="v2", name="Normalizer",
                 debug_mode=False, **kwargs):
        """initialise everything needed for this layer

        Args:
            strategy (str): in ["pic_bound", "seq_on_last", "inst_on_self"]
            csv_struct (str): the structure of the used data. in ["v2"]
            name (str): name of the layer. defaults to tha class name
            debug_mode (bool): activates some prints if enabled
        """
        # save strategy, structure and debug-state
        self.strategy = strategy
        self.debug_mode = debug_mode
        self.csv_struct = csv_struct

        # to properly work with numerical zeros
        self.eps = 1e-8

        if self.csv_struct == "v2":
            # structure object for csv-file-based-structure version 2
            self.arch = cpa.CSV_Architecture_v2()

        # base vector that will be used to genereate the vector per instance
        # which will be subtracted to normalize the hands
        self.substract_vec_base = tf.constant(
            np.array([0 for _ in range(len(self.arch.assignment))]),
            dtype=tf.float32
        )

        # initialise the swap pattern which is used by the normalize-on-self
        # approach to rearange the instance-tensors into theire original
        # order
        swap_pattern_np = np.concatenate(
            [
                self.arch.vector_intro_idxs,
                [self.arch.handedness_0_idx, self.arch.handedness_0_prob_idx],
                self.arch.hand_0_x,
                self.arch.hand_0_y,
                self.arch.hand_0_z,
                [self.arch.handedness_1_idx, self.arch.handedness_1_prob_idx],
                self.arch.hand_1_x,
                self.arch.hand_1_y,
                self.arch.hand_1_z,
            ]
        )
        self.swap_pattern = tf.constant(np.expand_dims(swap_pattern_np, -1))

        # init base layer configs
        super().__init__(name=name, **kwargs)

    def call(self, X):
        """heart of the layer: every batch will run through the graph that is
        created here

        Args:
            X (tf.Tensor): a given batch.
                           shape: [batch_size, seq_len, frame_repr_len]
        Returns:
            X (tf.Tensor): the batch normalized according to the given strategy
                           shape: [batch_size, seq_len, frame_repr_len]
        """
        # at this stage will be [None, sequence_len, csv_coordinate_dims],
        # where None represents the batch size.
        if self.debug_mode:
            tf.print("\nin normalizer")
            print("also eager in normalizer")

        # build the graphs, according to the given strategy
        if self.strategy == "pic_bound":
            X = self._pic_bound_transformer(X)
        elif self.strategy == "seq_on_last":
            X = tf.map_fn(
                fn=self._seq_on_last_transformer,
                elems=X
            )
        elif self.strategy == "inst_on_self":
            X = tf.map_fn(
                fn=(lambda t: tf.map_fn(
                    fn=self._inst_on_self_transformer,
                    elems=t
                )
                    ),
                elems=X
            )
        else:
            # signal if no strategy got triggered, possibly due to a wrongly
            # named strategy
            tf.print("\nWARNING! (Normalizer)\nNo strategy got triggered by" \
                     " given keyword.\nBatch is passed without normalization-" \
                     "transformation")
        return (X)

    def _pic_bound_transformer(self, X):
        """Strategy 1 (pic_bound):
        Normalize correlated to the x-y space of the given of the picture.
        -> This is just the output from mediapipe: from 0 to 1 for each
           dimension, related to the frame. That also means, for this strategy,
           nothing more needs to be done

        Args:
            X (tf.Tensor): a given batch.
                           shape: [batch_size, seq_len, frame_repr_len]

        Returns:
            X (tf.Tensor): just the same as the given X, nothing to do here
        """
        if self.debug_mode:
            tf.print(
                "nothing to do in mapped method '_pic_bound_transformer()'"
            )
        return (X)

    @tf.function
    def _seq_on_last_transformer(self, X):
        """Strategy 2 (seq_on_last):
        Normalize coordinates of the complete sequence (seperated per hand) to
        the wrist points of the most current point which is available for this
        hand

        Args:
            X (tf.Tensor): a sequence of the batch
                           shape: [seq_len, frame_repr_len]

        Returns:
            X (tf.Tensor): the sequence, normalized per hand, to the most
                           current, available wrist point
        """
        # NOTE: this approach is tightly bound to csv-structure 2. therefore
        # 		provide a print if anyone in the future will use this with
        # 		another csv-architcture
        if self.csv_struct != "v2":
            tf.print("\n\nWarning! Using Normalize-layer with\n" \
                     "_seq_on_last_transformer and another csv-structure\n" \
                     "than v2. This might lead to unexpacted behaviour when\n" \
                     "the wrist points are gathered! It is recommended to\n" \
                     "validate the layers behaviour and to look up the\n" \
                     "source code.")
        if self.debug_mode:
            tf.print("\n\n\n      in transformer: SEQ_ON_LAST      ")
            tf.print("-- FOR MAXIMUM INFORMATION, DISSABLE --\n" \
                     "-- @tf.function DECORATOR AND ENABLE --\n" \
                     "--          EAGER EXECUTION          --")

        # this reference is only used if a debug print is requested. assign
        # it here anyway as a conditional assignment only in debug mode violates
        # graph rules
        X_init = X

        # get the wrist coordiantes across all frames of the instance
        wrists_0 = tf.gather(
            X,
            [
                self.arch.hand_0_x[0],
                self.arch.hand_0_y[0],
                self.arch.hand_0_z[0]
            ],
            axis=-1
        )
        wrists_1 = tf.gather(
            X,
            [
                self.arch.hand_1_x[0],
                self.arch.hand_1_y[0],
                self.arch.hand_1_z[0]
            ],
            axis=-1
        )

        # DEPCRACATED approach
        # mask where hands were detected:
        # based on the std along the frame-wirsts-axis
        if False:
            wrists_0_stds = tf.math.reduce_std(wrists_0, axis=-1)
            wrists_1_stds = tf.math.reduce_std(wrists_1, axis=-1)
            wrists_0_mask = tf.greater(wrists_0_stds, self.eps)
            wrists_1_mask = tf.greater(wrists_1_stds, self.eps)

        # NEW approach
        # mask where hands were detected:
        # based on the [...]_pred-mark in the coordinate representation
        hand_0_pred = X[:, self.arch.handedness_0_idx]
        hand_1_pred = X[:, self.arch.handedness_1_idx]
        wrists_0_mask = tf.not_equal(
            hand_0_pred,
            self.arch.no_hand_detected_val
        )
        wrists_1_mask = tf.not_equal(
            hand_1_pred,
            self.arch.no_hand_detected_val
        )

        # get indexes of detected hands
        wrists_0_det = tf.where(wrists_0_mask)[:, 0]
        wrists_1_det = tf.where(wrists_1_mask)[:, 0]

        # get latest detected hand, predifine to enable seamless AutoGraph use
        # substract it's wrist coordinate values
        # go seperated path for both hands and concatenate later, as the
        # detection of one hand has to do nothing with the detection of the
        # other one
        first_valid_wrist_idx_0 = tf.constant(-1, dtype=tf.int64)
        substract_vec_0 = self.substract_vec_base
        # pre initialise for autograph compatiblity, even as the matrix later
        # gets created by scattering the subtaction vector
        # repeat also needs to be done here, as the seq-len is unknown during
        # graph creation
        subtraction_matrix_0 = tf.repeat(
            tf.expand_dims(substract_vec_0, axis=0),
            repeats=tf.shape(X)[0],
            axis=0
        )
        # repeat the steps above for hand 1
        first_valid_wrist_idx_1 = tf.constant(-1, dtype=tf.int64)
        substract_vec_1 = self.substract_vec_base
        subtraction_matrix_1 = tf.repeat(
            tf.expand_dims(substract_vec_1, axis=0),
            repeats=tf.shape(X)[0],
            axis=0
        )

        # print first useful information in debug mode
        if self.debug_mode:
            tf.print("\nprocessed sequence (squeezed display):\n", X[:, 3:])
            tf.print("shape: ", X.shape)
            tf.print("positions_valid_hands hand_0: ", wrists_0_det)
            tf.print("positions_valid_hands hand_1: ", wrists_1_det)

        # only enter if there is at least one valid hand. build the subtraction
        # matrix and subtract it from the instance
        # NOTE: build the two matrices - one for each hand - based on a
        #		0-initialised matrix and add them together to the final
        #		subtraction matrix
        # hand_0
        if tf.not_equal(tf.shape(wrists_0_det)[0], 0):
            # reinitialise with same datatype as at local definition
            first_valid_wrist_idx_0 = wrists_0_det[-1]
            ref_wrist = tf.expand_dims(
                wrists_0[first_valid_wrist_idx_0],
                axis=1
            )
            # combine the indexes according to the dimension
            idx_s_comb = ([
                self.arch.hand_0_x,
                self.arch.hand_0_y,
                self.arch.hand_0_z
            ])
            # build the vector that shall be substracted from every valid hand
            subtraction_matrix_0 = self._get_sub_matrix_seq_on_last(
                ref_wrist,
                idx_s_comb,
                substract_vec_0,
                wrists_0_det,
                tf.shape(X)
            )
            if self.debug_mode:
                tf.print("CREATED sub_matr for hand_0")
        else:
            if self.debug_mode:
                tf.print("NO sub_matr created for hand_0, due to no valid " \
                         "hand in seq")
        # hands_1
        if tf.not_equal(tf.shape(wrists_1_det)[0], 0):
            # reinitialise with same datatype as at local definition
            first_valid_wrist_idx_1 = wrists_1_det[-1]
            ref_wrist = tf.expand_dims(
                wrists_1[first_valid_wrist_idx_1],
                axis=1
            )
            # combine the indexes according to the dimension
            idx_s_comb = ([
                self.arch.hand_1_x,
                self.arch.hand_1_y,
                self.arch.hand_1_z
            ])
            # build the vector that shall be substracted from every valid hand
            subtraction_matrix_1 = self._get_sub_matrix_seq_on_last(
                ref_wrist,
                idx_s_comb,
                substract_vec_1,
                wrists_1_det,
                tf.shape(X)
            )
            if self.debug_mode:
                tf.print("CREATED sub_matr for hand_1")
        else:
            if self.debug_mode:
                tf.print("NO sub_matr created for hand_1, due to no valid " \
                         "hand in seq")

        # combine the subtraction matrices that were created seperatly per hand,
        # but across all frames representated by the instance
        subtraction_matrix = tf.math.add(
            subtraction_matrix_0,
            subtraction_matrix_1
        )

        if self.debug_mode:
            tf.print("\nhand_0 snip BEFORE trans")
            tf.print(X[:, 4:9])

        # finally process the instance
        X = tf.math.subtract(X, subtraction_matrix)

        if self.debug_mode:
            tf.print("\nhand_0 snip AFTER trans")
            tf.print(X[:, 4:9])

        # this print will show a hugh output, but it will only print if the
        # @tf.function-decorator is uncommented and the model runs eagerly
        if self.debug_mode:
            print("\ninput sequence: ")
            print(X_init)
            print("\ntransformed sequence: ")
            print(X)
            print("\nused subtraction matrix: ")
            print(subtraction_matrix)

        return (X)

    @tf.function
    def _inst_on_self_transformer(self, X_t):
        """Strategy 3 (inst_on_self):
        Normalize all coordinates related to themselves, sothat they appear
        stational. Therefore, each framerepresentation needs to be normalized
        in relation to itself, splitted by hands and dimensions

        FLOW:
        - extract both hand's coordinate's seperate
        - check if a valid hand is there
        - normalize on wrist per hand
        - restack the normalized hands with the other parameters to the return
          value

        Args:
            X (tf.Tensor): a fram representation of a sequence of the batch
                           shape: [frame_repr_len]

        Returns:
            X (tf.Tensor): the representation, normalized per hand and
                           dimension, but only for valid hands (do nothing for
                           not detected hands)
        """
        # NOTE: the input is every frame representation itself and not the
        # 		whole instance (which would mean be sequence of representations)
        # NOTE: this approach is tightly bound to csv-structure 2. therefore
        # 		provide a print if anyone in the future will use this with
        # 		another csv-architcture
        if self.csv_struct != "v2":
            tf.print("\n\nWarning! Using Normalize-layer with\n" \
                     "_inst_on_self_transformer and another csv-structure\n" \
                     "than v2. This might lead to unexpacted behaviour when\n" \
                     "the values are restacked! It is recommended to\n" \
                     "validate the layers behaviour and to look up the\n" \
                     "source code.")
        if self.debug_mode:
            tf.print("\n\n\n      in transformer: INST_ON_SELF      ")
            tf.print("-- FOR MAXIMUM INFORMATION, DISSABLE --\n" \
                     "-- @tf.function DECORATOR AND ENABLE --\n" \
                     "--          EAGER EXECUTION          --")

        # this reference is only used if a debug print is requested. assign
        # it here anyway as a conditional assignment only in debug mode violates
        # graph rules
        X_init = X_t

        # get the handedness prediction per hand
        X_t_hand_0_pred = X_t[self.arch.handedness_0_idx]
        X_t_hand_1_pred = X_t[self.arch.handedness_1_idx]

        # assign some stuff here, to not do it within the conditions, to easaly
        # and smoothly enable autograph
        # collect the coordinates per dimension per hand
        # init values
        X_t_init_vals = tf.gather(
            X_t,
            self.arch.vector_intro_idxs.tolist(),
            axis=-1
        )
        # hand_0
        X_t_hand_0_x = tf.gather(
            X_t,
            self.arch.hand_0_x.tolist(),
            axis=-1
        )
        X_t_hand_0_y = tf.gather(
            X_t,
            self.arch.hand_0_y.tolist(),
            axis=-1
        )
        X_t_hand_0_z = tf.gather(
            X_t,
            self.arch.hand_0_z.tolist(),
            axis=-1
        )
        # hand_1
        X_t_hand_1_x = tf.gather(
            X_t,
            self.arch.hand_1_x.tolist(),
            axis=-1
        )
        X_t_hand_1_y = tf.gather(
            X_t,
            self.arch.hand_1_y.tolist(),
            axis=-1
        )
        X_t_hand_1_z = tf.gather(
            X_t,
            self.arch.hand_1_z.tolist(),
            axis=-1
        )

        # print first useful information in debug mode
        if self.debug_mode:
            tf.print(
                "\nprocessed frame representation (squeezed display):\n",
                X_t
            )
            tf.print("shape: ", X_t.shape)
            tf.print("handedness_pred hand_0: ", X_t_hand_0_pred)
            tf.print("handedness_pred hand_1: ", X_t_hand_1_pred)

        # hand_0: modifiy if there is a valid hand
        if tf.not_equal(X_t_hand_0_pred, self.arch.no_hand_detected_val):
            if self.debug_mode:
                tf.print("IN for hand_0: normalize coordinates per dimension")
            # normalize all dimensions by themselves. use the preinitialised
            # tensors
            X_t_hand_0_x = self._min_max_0_1(X_t_hand_0_x)
            X_t_hand_0_y = self._min_max_0_1(X_t_hand_0_y)
            X_t_hand_0_z = self._min_max_0_1(X_t_hand_0_z)
        else:
            if self.debug_mode:
                tf.print("NOT_in for hand_0: no valid hand detected, pass hand")
        # hand_1: modifiy if there is a valid hand
        if tf.not_equal(X_t_hand_1_pred, self.arch.no_hand_detected_val):
            if self.debug_mode:
                tf.print("IN for hand_1: normalize coordinates per dimension")
            # normalize all dimensions by themselves. use the preinitialised
            # tensors
            X_t_hand_1_x = self._min_max_0_1(X_t_hand_1_x)
            X_t_hand_1_y = self._min_max_0_1(X_t_hand_1_y)
            X_t_hand_1_z = self._min_max_0_1(X_t_hand_1_z)
        else:
            if self.debug_mode:
                tf.print("NOT_in for hand_1: no valid hand detected, pass hand")

        # this print will show a hugh output, but it will only print if the
        # @tf.function-decorator is uncommented and the model runs eagerly
        if self.debug_mode:
            print("\ninitial look of the frame representation:\n")
            print(X_t)

        # restack the instance tensor with the possibly modified sequences
        X_t = tf.concat(
            [
                X_t_init_vals,
                tf.expand_dims(X_t[self.arch.handedness_0_idx], 0),
                tf.expand_dims(X_t[self.arch.handedness_0_prob_idx], 0),
                X_t_hand_0_x,
                X_t_hand_0_y,
                X_t_hand_0_z,
                tf.expand_dims(X_t[self.arch.handedness_1_idx], 0),
                tf.expand_dims(X_t[self.arch.handedness_1_prob_idx], 0),
                X_t_hand_1_x,
                X_t_hand_1_y,
                X_t_hand_1_z
            ],
            0
        )

        # this print will show a hugh output, but it will only print if the
        # @tf.function-decorator is uncommented and the model runs eagerly
        if self.debug_mode:
            print("\nrestacked instance with possibly normalized values:\n")
            print(X_t)
            print("\ncurr DIFF to init-inst (for val):")
            print(" NOTE: as the values got normalized, the diff tensor is" \
                  "\n not equal to zero, despite that values are at the right" \
                  "\n index. Deactivate the normalize-function calls within" \
                  "\n this layers function, to validate the correct swapping")
            print(tf.subtract(X_init, X_t))

        # swap the values back into original order
        shape = tf.shape(X_t, out_type=tf.int64)
        X_t = tf.scatter_nd(self.swap_pattern, X_t, shape)

        # this print will show a hugh output, but it will only print if the
        # @tf.function-decorator is uncommented and the model runs eagerly
        if self.debug_mode:
            print("\ntensor after scattering with swap pattern (this one\n" \
                  "is finally returned")
            print(X_t)
            print("\ncurr diff to init inst (for val):")
            print(" NOTE: as the values got normalized, the diff tensor is" \
                  "\n not equal to zero, despite that values are at the right" \
                  "\n index. Deactivate the normalize-function calls within" \
                  "\n this layers function, to validate the correct swapping")
            print(tf.subtract(X_init, X_t))

        return (X_t)

    def _min_max_0_1(self, X_seq):
        """normalize a given sequence from X on itself (min/max to 0/1)

        Args:
            X_seq (tf.Tensor): a sequence to normalize onto 0-1. While writing,
                               the input is always a slice from one hand and
                               one dimension, e.g. all x-coordinates from hand_0
                               shape: [seq_len]

        Returns:
            X_seq (tf.Tensor): the normalized input sequence
        """
        X_seq = tf.math.divide(
            tf.math.subtract(
                X_seq,
                tf.math.reduce_min(X_seq)
            ),
            tf.add(
                tf.math.subtract(
                    tf.math.reduce_max(X_seq),
                    tf.math.reduce_min(X_seq)
                ),
                self.eps
            )
        )
        return (X_seq)

    def _get_sub_matrix_seq_on_last(self, ref_wrist, idx_s_comb,
                                    substract_vec, wrists_det, X_shape):
        """provide the matrix which is used by the _seq_on_last_transformer-
        function to normalize a sequence of frame representations. the function
        builds a matrix, which's vectors repeat the x,y&z-coordinate of the
        wrist point from the most current hand. The coordinates get placed at
        the right indexes for dimension&hand; unvalid hands are also already
        concidered and simply reciev zeros.
        The function is called one per hand and the returned matrix will contain
        zeros for the other hand.
        The matrices genereated that way, will than be added, and the resulting
        matrix can simply get subracted from the instance (which is done in
        the _seq_on_last_transformer itself)

        Args:
            ref_wrist (tf.Tensor): the wirst coordinates (x,y,z) of the wrist
                                   point on which the whole sequence shall be
                                   normalized
            idx_s_comb (list(np.array)): contains three numpy arrays, which
                                         contain all indexes for one dimension
                                         in the current frame representation
            substract_vec (tf.Tensor): this vector will contain the x,y and z
                                       values at the right indexes, for one
                                       hand. This vector is to be subtracted
                                       from valid hands
            wrists_det (tf.Tensor): contains the indexes of the validly detected
                                    hands. It is needed to know how often and
                                    where the substract_vec needs to be inserted
            X_shape (tf.): shape of an instance, which is
                           [seq_len, frame_repr_len]. The shape is needed to
                           create the subtraction_matrix, as the tf.scatter_nd-
                           function wants an explicit shape

        Returns:
            subtraction_matrix (tf.Tensor): the matrix which contains the right
                                            values at the right position to
                                            normalize one hand of an instance.
                                            To normalize the whole instance,
                                            this function needs to be called per
                                            hand, the recieved matrices combined
                                            and this final matrix can then be
                                            subtracted from the whole instance
        """
        # loop through the coordinate points. work with return values according
        # to the autograph guidlines
        subtract_vec = substract_vec
        # itterate over x, y and z dimension. NOTE: cannot use zip function
        # with autograph, therefore use index assignment
        for i in range(len(idx_s_comb)):
            # assign the x/y/z-value which is used for normalization across all
            # frame representations, as well as the indexes for each frame that
            # shall be affected by it
            ref_val, idx_s = ref_wrist[i], idx_s_comb[i]

            # expand dims to properly use the array with tf's scatter_nd
            # function
            val_idx = tf.convert_to_tensor(
                np.expand_dims(idx_s, 1),
                dtype=tf.int32
            )

            # create repeated vec that will be the coordinat of the current
            # wrist point) whichs contains the coordinate the number of times
            # it will be imputed into the substraction vector
            rep_val = tf.repeat(ref_val, repeats=tf.shape(idx_s)[0])

            # scattered inputs
            subtract_vec = tf.tensor_scatter_nd_update(
                subtract_vec,
                val_idx,
                rep_val
            )

        # scatter the vector according to the instance size and according to
        # the properly detected hands
        rep_sub_vec = tf.repeat(
            tf.expand_dims(subtract_vec, axis=0),
            repeats=tf.shape(wrists_det)[0],
            axis=0
        )

        # create the subtraction matrix for the given hand. NOTE: this matrix
        # needs to be combined with the one for the other hand, before it is
        # finally applied to the instance
        subtraction_matrix = tf.scatter_nd(
            tf.expand_dims(wrists_det, axis=1),
            rep_sub_vec,
            shape=tf.cast(X_shape, tf.int64)
        )
        return (subtraction_matrix)

    def compute_output_shape(self, input_shape):
        """this function is needed to seemingly use the custom layer

        Args:
            input_shape
        Returns:
            input_shape
        """
        return (input_shape)

    def get_config(self):
        """this function is needed to seemingly use the custom layer

        Returns:
            _configs (dict): an assignment of all variables that were used to
                             initialise the layer
        """
        config = {
            "strategy": self.strategy,
            "csv_struct": self.csv_struct,
            "debug_mode": self.debug_mode
        }
        base_config = super().get_config()
        return (dict(list(base_config.items()) + list(config.items())))


# decorator ensures easy loading of the custom layers
@keras.utils.register_keras_serializable(package="Custom", name="Passer")
class Passer(keras.layers.Layer):
    """use this empty layer to insert it during developement at any point in
    the model to see how the tensor look like, test operations, or do whatever
    """

    def __init__(self, print_arg="dummy", name="Passer", **kwargs):
        """initialise everything needed for this layer

        Args:
            print_arg (str): use this string if something specific shall be
                             printed by Passer-Layers, placed at different
                             points in the model
            name (str): name of the layer. defaults to tha class name. needs
                        to be specified if the layer is used multiple times, to
                        maintain uniqueness. Otherwise the API will throw an
                        error
        """
        self.print_arg = print_arg
        # init base layer configs
        super().__init__(name=name, **kwargs)

    def call(self, X):
        """heart of the layer: use this for whatever

        Args:
            X (tf.Tensor): a given batch.
                           shape: [batch_size, seq_len, frame_repr_len]
        Returns:
            X (tf.Tensor): a given batch.
                           shape: [batch_size, seq_len, frame_repr_len]
        """
        # play around here
        
        print("\nin passer")
        print(self.print_arg)
        print(X)
        print(X.shape)
        print("out passer")
        
        return (X)

    def compute_output_shape(self, input_shape):
        """this function is needed to seemingly use the custom layer

        Args:
            input_shape
        Returns:
            input_shape
        """
        return (input_shape)

    def get_config(self):
        """this function is needed to seemingly use the custom layer

        Returns:
            _configs (dict): an assignment of all variables that were used to
                             initialise the layer
        """
        config = {
            "print_arg": self.print_arg,
        }
        base_config = super().get_config()
        return (dict(list(base_config.items()) + list(config.items())))


if __name__ == "__main__":
    pass
