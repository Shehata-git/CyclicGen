"""Train a voxel flow model on ucf101 dataset with optimizations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dataset
import numpy as np
import os
import tensorflow as tf
from datetime import datetime
from CyclicGen_model import Voxel_flow_model
from utils.image_utils import imwrite
from skimage.measure import compare_ssim as ssim
from vgg16 import Vgg16

# Constants for better readability and maintenance
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 3
EDGE_CHANNELS = 1

FLAGS = tf.app.flags.FLAGS

# Define necessary FLAGS
tf.app.flags.DEFINE_string('train_dir', './CyclicGen_checkpoints_stage1/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_image_dir', './voxel_flow_train_image/',
                           """Directory where to output images.""")
tf.app.flags.DEFINE_string('test_image_dir', './voxel_flow_test_image_baseline/',
                           """Directory where to output images.""")
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', None,
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_integer('max_steps', 10000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer(
    'batch_size', 8, 'The number of samples in each batch.')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0001,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_integer('training_data_step', 1, """The step used to reduce training data size""")
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 2000, """Steps between checkpoint saves""")
tf.app.flags.DEFINE_integer('summary_steps', 100, """Steps between summary writes""")
tf.app.flags.DEFINE_integer('shuffle_buffer_size', 10000, """Buffer size for shuffling data""")
tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.9, """Fraction of GPU memory to use""")


def _parse_and_preprocess(filename, seed=1):
    """Combined parsing and preprocessing function for dataset efficiency."""
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string, channels=3)
    # Normalize to [-1, 1] range
    image_normalized = tf.cast(image_decoded, dtype=tf.float32) / 127.5 - 1.0
    
    # Apply augmentations
    image_flipped_lr = tf.image.random_flip_left_right(image_normalized, seed=seed)
    image_flipped_ud = tf.image.random_flip_up_down(image_flipped_lr, seed=seed)
    image_cropped = tf.random_crop(image_flipped_ud, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS], seed=seed)
    
    return image_cropped


def create_dataset(data_list, training_data_step, batch_size, is_training=True):
    """Create a TensorFlow dataset with optimized operations."""
    # Use subset of data if specified
    filtered_data_list = data_list[::training_data_step]
    
    # Create dataset from filenames
    dataset = tf.data.Dataset.from_tensor_slices(tf.constant(filtered_data_list))
    
    if is_training:
        # Shuffle and repeat for training
        dataset = dataset.shuffle(
            buffer_size=FLAGS.shuffle_buffer_size, 
            seed=1, 
            reshuffle_each_iteration=True
        ).repeat()
    
    # Apply preprocessing with parallelism
    dataset = dataset.map(
        lambda filename: _parse_and_preprocess(filename), 
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset


def train(dataset_frame1, dataset_frame2, dataset_frame3):
    """Trains a model with optimizations."""
    with tf.Graph().as_default():
        # Read data lists
        data_list_frame1 = dataset_frame1.read_data_list_file()
        data_list_frame2 = dataset_frame2.read_data_list_file()
        data_list_frame3 = dataset_frame3.read_data_list_file()
        
        # Create optimized datasets
        dataset1 = create_dataset(data_list_frame1, FLAGS.training_data_step, FLAGS.batch_size)
        dataset2 = create_dataset(data_list_frame2, FLAGS.training_data_step, FLAGS.batch_size)
        dataset3 = create_dataset(data_list_frame3, FLAGS.training_data_step, FLAGS.batch_size)
        
        # Create iterators
        batch_frame1 = dataset1.make_initializable_iterator()
        batch_frame2 = dataset2.make_initializable_iterator()
        batch_frame3 = dataset3.make_initializable_iterator()

        # Get input tensors
        input1 = batch_frame1.get_next()
        input2 = batch_frame2.get_next()
        input3 = batch_frame3.get_next()

        # Create edge detection
        with tf.variable_scope("EdgeDetection"):
            edge_vgg_1 = Vgg16(input1, reuse=None)
            edge_vgg_3 = Vgg16(input3, reuse=True)

            edge_1 = tf.nn.sigmoid(edge_vgg_1.fuse)
            edge_3 = tf.nn.sigmoid(edge_vgg_3.fuse)

            # Reshape edges to match input dimensions
            edge_1 = tf.reshape(edge_1, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, EDGE_CHANNELS])
            edge_3 = tf.reshape(edge_3, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, EDGE_CHANNELS])

        # Create model and compute loss
        with tf.variable_scope("Cycle_DVF"):
            model1 = Voxel_flow_model()
            
            # Combine inputs with edges
            model_input = tf.concat([input1, input3, edge_1, edge_3], 3)
            
            # Run inference
            prediction1, flow1 = model1.inference(model_input)
            
            # Calculate loss
            reproduction_loss1 = model1.l1loss(prediction1, input2)

        # Get trainable variables, excluding those from edge detection
        t_vars = tf.trainable_variables()
        print('All model layers:')
        for var in t_vars: 
            print(var.name)
            
        dof_vars = [var for var in t_vars if not 'hed' in var.name]
        print('Optimizing layers:')
        for var in dof_vars: 
            print(var.name)

        # Total loss
        total_loss = reproduction_loss1

        # Create optimizer with gradient clipping for stability
        global_step = tf.train.get_or_create_global_step()
        learning_rate = FLAGS.initial_learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        
        # Use gradient clipping to prevent exploding gradients
        gradients, variables = zip(*optimizer.compute_gradients(total_loss, var_list=dof_vars))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        update_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

        # Create and group summaries
        with tf.name_scope("summaries"):
            tf.summary.scalar('total_loss', total_loss)
            tf.summary.image('input1', input1, 3)
            tf.summary.image('input2', input2, 3)
            tf.summary.image('input3', input3, 3)
            tf.summary.image('edge_1', edge_1, 3)
            tf.summary.image('edge_3', edge_3, 3)
            tf.summary.image('prediction1', prediction1, 3)
            
            # Add gradient norm summaries
            for grad, var in zip(gradients, variables):
                if grad is not None:
                    tf.summary.histogram(f"{var.name}/gradient", grad)
            
            summary_op = tf.summary.merge_all()

        # Create a saver with improved keep_checkpoint_every_n_hours
        saver = tf.train.Saver(
            max_to_keep=10,
            keep_checkpoint_every_n_hours=2.0
        )

        # Configure GPU memory usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction

        # Create session
        with tf.Session(config=config) as sess:
            # Initialize or restore model
            if FLAGS.pretrained_model_checkpoint_path:
                restorer = tf.train.Saver()
                restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
                print('%s: Pre-trained model restored from %s' %
                    (datetime.now(), FLAGS.pretrained_model_checkpoint_path))
                sess.run([batch_frame1.initializer, batch_frame2.initializer, batch_frame3.initializer])
            else:
                # Initialize variables and datasets
                init_op = tf.group(
                    tf.global_variables_initializer(),
                    tf.local_variables_initializer()
                )
                sess.run([init_op, batch_frame1.initializer, batch_frame2.initializer, batch_frame3.initializer])

            # Restore HED edge detection model
            meta_model_file = 'hed_model/new-model.ckpt'
            saver2 = tf.train.Saver(var_list=[v for v in tf.global_variables() if "hed" in v.name])
            saver2.restore(sess, meta_model_file)

            # Setup summary writer
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=sess.graph)

            # Calculate epoch information for logging
            data_size = len(data_list_frame1)
            epoch_steps = int(data_size / FLAGS.batch_size)
            
            # Print training configuration
            print(f"Training with {data_size} samples")
            print(f"Steps per epoch: {epoch_steps}")
            print(f"Batch size: {FLAGS.batch_size}")

            # Training loop
            last_epoch = -1
            for step in range(0, FLAGS.max_steps):
                epoch = step // epoch_steps
                
                # Print epoch transition
                if epoch > last_epoch:
                    last_epoch = epoch
                    print(f'Starting Epoch {epoch}')

                # Run single training step
                try:
                    _, loss_value = sess.run([update_op, total_loss])
                except tf.errors.OutOfRangeError:
                    # Reinitialize iterators if we run out of data
                    sess.run([batch_frame1.initializer, batch_frame2.initializer, batch_frame3.initializer])
                    continue

                # Log losses periodically
                if step % 10 == 0:
                    print(f"Step {step}, Epoch {epoch}, Loss: {loss_value:.6f}")

                # Output summary
                if step % FLAGS.summary_steps == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                # Save checkpoint
                if step % FLAGS.save_checkpoint_steps == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    print(f"Model saved at step {step}")


def test(dataset_frame1, dataset_frame2, dataset_frame3):
    """Perform optimized testing on a trained model."""
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    with tf.Graph().as_default():
        # Create input and target placeholders
        input_placeholder = tf.placeholder(tf.float32, shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, 6))
        target_placeholder = tf.placeholder(tf.float32, shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

        # Edge detection
        with tf.variable_scope("EdgeDetection"):
            edge_vgg_1 = Vgg16(input_placeholder[:, :, :, :3], reuse=None)
            edge_vgg_3 = Vgg16(input_placeholder[:, :, :, 3:6], reuse=True)

            edge_1 = tf.nn.sigmoid(edge_vgg_1.fuse)
            edge_3 = tf.nn.sigmoid(edge_vgg_3.fuse)

            edge_1 = tf.reshape(edge_1, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, EDGE_CHANNELS])
            edge_3 = tf.reshape(edge_3, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, EDGE_CHANNELS])

        # Create model
        with tf.variable_scope("Cycle_DVF"):
            model = Voxel_flow_model(is_train=False)
            prediction, warped_outputs = model.inference(tf.concat([input_placeholder, edge_1, edge_3], 3))

        # Configure GPU memory usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        # Create session
        with tf.Session(config=config) as sess:
            # Restore model
            if not FLAGS.pretrained_model_checkpoint_path:
                raise ValueError("Must specify pretrained model checkpoint path for testing")
                
            restorer = tf.train.Saver()
            restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
            print('%s: Pre-trained model restored from %s' %
                (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

            # Process test dataset
            data_list_frame1 = dataset_frame1.read_data_list_file()
            data_size = len(data_list_frame1)
            data_list_frame2 = dataset_frame2.read_data_list_file()
            data_list_frame3 = dataset_frame3.read_data_list_file()

            total_psnr = 0.0
            total_ssim = 0.0
            valid_frames = 0

            print(f"Testing on {data_size} samples")
            
            # Process each test sample
            for id_img in range(0, data_size):
                UCF_index = data_list_frame1[id_img][:-12]
                
                # Prepare file paths with cleaner string formatting
                frame1_path = os.path.join('ucf101_interp_ours', f"{data_list_frame1[id_img][:-5]}00.png")
                frame2_path = os.path.join('ucf101_interp_ours', f"{data_list_frame2[id_img][:-5]}01_gt.png")
                frame3_path = os.path.join('ucf101_interp_ours', f"{data_list_frame3[id_img][:-5]}02.png")
                mask_path = os.path.join('motion_masks_ucf101_interp', f"{data_list_frame3[id_img][:-11]}motion_mask.png")
                
                # Load data
                batch_data_frame1 = [dataset_frame1.process_func(frame1_path)]
                batch_data_frame2 = [dataset_frame2.process_func(frame2_path)]
                batch_data_frame3 = [dataset_frame3.process_func(frame3_path)]
                batch_data_mask = [dataset_frame3.process_func(mask_path)]

                # Convert to numpy arrays
                batch_data_frame1 = np.array(batch_data_frame1)
                batch_data_frame2 = np.array(batch_data_frame2)
                batch_data_frame3 = np.array(batch_data_frame3)
                batch_data_mask = (np.array(batch_data_mask) + 1.0) / 2.0

                # Run model inference
                feed_dict = {
                    input_placeholder: np.concatenate((batch_data_frame1, batch_data_frame3), 3),
                    target_placeholder: batch_data_frame2
                }
                
                prediction_np, target_np, warped_img1, warped_img2 = sess.run(
                    [prediction, target_placeholder, model.warped_img1, model.warped_img2],
                    feed_dict=feed_dict
                )

                # Save prediction
                output_dir = os.path.join('ucf101_interp_ours', str(UCF_index))
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, 'frame_01_CyclicGen.png')
                imwrite(output_path, prediction_np[0][-1, :, :, :])
                
                # Calculate metrics for areas with motion
                mask_sum = np.sum(batch_data_mask)
                if mask_sum > 0:
                    # Apply mask to predictions and targets
                    img_pred_mask = np.expand_dims(batch_data_mask[0], -1) * (prediction_np[0][-1] + 1.0) / 2.0
                    img_target_mask = np.expand_dims(batch_data_mask[0], -1) * (target_np[-1] + 1.0) / 2.0
                    
                    # Calculate PSNR
                    mse = np.sum((img_pred_mask - img_target_mask) ** 2) / (3. * mask_sum)
                    psnr_cur = 20.0 * np.log10(1.0) - 10.0 * np.log10(mse)
                    
                    # Calculate SSIM
                    img_pred_gray = rgb2gray((prediction_np[0][-1] + 1.0) / 2.0)
                    img_target_gray = rgb2gray((target_np[-1] + 1.0) / 2.0)
                    ssim_cur = ssim(img_pred_gray, img_target_gray, data_range=1.0)
                    
                    # Accumulate metrics
                    total_psnr += psnr_cur
                    total_ssim += ssim_cur
                    valid_frames += 1
                    
                    print(f"Sample {id_img+1}/{data_size}: PSNR={psnr_cur:.4f}, SSIM={ssim_cur:.4f}")
                else:
                    print(f"Sample {id_img+1}/{data_size}: No valid motion mask, skipping metrics")
                    
            # Report final metrics
            if valid_frames > 0:
                avg_psnr = total_psnr / valid_frames
                avg_ssim = total_ssim / valid_frames
                print(f"Overall PSNR: {avg_psnr:.4f} dB")
                print(f"Overall SSIM: {avg_ssim:.4f}")
                
                # Create a simple results file
                with open(os.path.join(FLAGS.test_image_dir, 'results.txt'), 'w') as f:
                    f.write(f"Test Results\n")
                    f.write(f"====================\n")
                    f.write(f"Total samples: {data_size}\n")
                    f.write(f"Valid samples: {valid_frames}\n")
                    f.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
                    f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            else:
                print("No valid frames found for evaluation")


if __name__ == '__main__':
    # Ensure output directories exist
    os.makedirs(FLAGS.train_dir, exist_ok=True)
    os.makedirs(FLAGS.train_image_dir, exist_ok=True)
    os.makedirs(FLAGS.test_image_dir, exist_ok=True)

    if FLAGS.subset == 'train':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        data_list_path_frame1 = "data_list/ucf101_train_files_frame1.txt"
        data_list_path_frame2 = "data_list/ucf101_train_files_frame2.txt"
        data_list_path_frame3 = "data_list/ucf101_train_files_frame3.txt"

        ucf101_dataset_frame1 = dataset.Dataset(data_list_path_frame1)
        ucf101_dataset_frame2 = dataset.Dataset(data_list_path_frame2)
        ucf101_dataset_frame3 = dataset.Dataset(data_list_path_frame3)

        print("Starting training pipeline...")
        train(ucf101_dataset_frame1, ucf101_dataset_frame2, ucf101_dataset_frame3)

    elif FLAGS.subset == 'test':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Changed from empty string to use GPU for testing

        data_list_path_frame1 = "data_list/ucf101_test_files_frame1.txt"
        data_list_path_frame2 = "data_list/ucf101_test_files_frame2.txt"
        data_list_path_frame3 = "data_list/ucf101_test_files_frame3.txt"

        ucf101_dataset_frame1 = dataset.Dataset(data_list_path_frame1)
        ucf101_dataset_frame2 = dataset.Dataset(data_list_path_frame2)
        ucf101_dataset_frame3 = dataset.Dataset(data_list_path_frame3)

        print("Starting testing pipeline...")
        test(ucf101_dataset_frame1, ucf101_dataset_frame2, ucf101_dataset_frame3)
    else:
        print(f"Unknown subset: {FLAGS.subset}")
