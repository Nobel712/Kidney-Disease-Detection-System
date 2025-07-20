def gradCam(image, true_label, layer_conv_name):
    model_grad = tf.keras.models.Model(inputs = m.input, 
                                  outputs = [m.get_layer(layer_conv_name).output, 
                                             m.output])
    with tf.GradientTape() as tape:
        conv_output, predictions = model_grad(image)
        tape.watch(conv_output)
        loss = tf.losses.binary_crossentropy(true_label, predictions)
    grad = tape.gradient(loss, conv_output)
    grad = K.mean(tf.abs(grad), axis = (0, 1, 2))
    conv_output = np.squeeze(conv_output.numpy())
    for i in range(conv_output.shape[-1]):
        conv_output[:,:, i] = conv_output[:,:, i]*grad[i]
    heatmap = tf.reduce_mean(conv_output, axis = -1)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap/tf.reduce_max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    return np.squeeze(heatmap), np.squeeze(image)

def getHeatMap(images, labels):
    heatmaps = []
    for index in range(128):
        heatmap, image = gradCam(images[index: index + 1], 
                                               labels[index: index + 1], 
                                           'relu')
        heatmaps.append(heatmap)
    return np.array(heatmaps)
  
heatmaps = getHeatMap(images, labels)
heatmaps.shape
