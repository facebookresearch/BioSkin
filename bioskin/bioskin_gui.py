# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from PyQt5.QtCore import (Qt, QRectF, QEvent, QObject, QPoint)
from PyQt5.QtWidgets import (QApplication, QStatusBar, QDesktopWidget, QMainWindow, QDockWidget, QGridLayout, QGroupBox,
                             QLabel, QProgressBar, QPushButton, QSlider, QTabWidget, QMenu, QAction, QVBoxLayout,
                             QWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFrame,
                             QGraphicsDropShadowEffect)
from PyQt5.QtGui import (QImage, QPixmap, QBrush, QColor, QFont, QPainter, QKeySequence, QIcon, QPainterPath, QPen)
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'
import sys
import cv2
import torch
import numpy as np
import qdarkstyle
import time
import argparse
from bioskin.bioskin import BioSkinInference
from bioskin.character import Character
import bioskin.utils.io as io
import bioskin.spectrum.color_spectrum as color
from bioskin.parameters.parameter import SKIN_PROPS
from bioskin.parameters.parameter import SKIN_MIN_VALUES
from bioskin.parameters.parameter import SKIN_MAX_VALUES
from bioskin.reconstruct_in_batches import reconstruct_in_batches_with_ops

pixelsize = 20


def parse_arguments():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='GUI for interactive estimation and manipulation '
                                                 'of skin properties from diffuse reflectance')

    parser.add_argument('--path_to_model', nargs='?',
                        type=str, default='../pretrained_models/',
                        help='Folder to Json files with pretrained nn models')

    parser.add_argument('--json_model', nargs='?',
                        type=str, default='BioSkinAO',
                        help='Json file with trained nn model')

    parser.add_argument('--input_folder', nargs='?',
                        type=str, default='input_images/',
                        help='Input Folder with diffuse reflectance to reconstruct')
    parser.add_argument('--output_folder', nargs='?',
                        type=str, default='reconstruction_demo/',
                        help='Output Folder with reconstruction and editing results')

    parser.add_argument('--max_width', nargs='?',
                        type=int, default=2048,
                        help='Maximum image width for interactive demo. '
                             'Edited images can still be stored at the original resolution')

    args = parser.parse_args()
    return args


def pixmap_from_array(array):
    array = color.linear_to_sRGB(array)
    array = np.clip(array, 0, 1)
    scaled = np.uint8(np.round(array * 255))
    height, width, channel = scaled.shape
    image = QImage(scaled.data, width, height, channel * width, QImage.Format_BGR888)
    return QPixmap(image)


class ZoomView(QGraphicsView):
    def __init__(self, parent=None):
        QGraphicsView.__init__(self, parent=parent)
        # main view zoom level
        self.zoom = 0
        # main view display texture
        self.photo = QGraphicsPixmapItem()

        # interaction
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        # looks
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setFrameShape(QFrame.NoFrame)

        # key to toggle edited/original
        self.toggleKey = Qt.Key_E
        #key to toggle diffuse/diffuse+specular
        self.toggleKey2 = Qt.Key_D

    def wheelEvent(self, event: QEvent):
        if event.angleDelta().y() > 0:
            if self.zoom > 15:
                return
            factor = 1.25
            self.zoom += 1
        else:
            if self.zoom < -1:
                return
            factor = 0.8
            self.zoom -= 1
        self.scale(factor, factor)

    def updateView(self, pixmap: QPixmap, pixmap_orig: QPixmap):
        # font (top)
        f_top = QFont()
        text_color = QColor(Qt.white)

        f_top.setPixelSize(pixelsize)
        f_top.setBold(True)
        offset = 50

        # font (top)
        f_bottom = QFont()
        f_bottom.setPixelSize(pixelsize - 5)

        # set up QGraphicsView scene
        key = QKeySequence(self.toggleKey).toString()
        self.photo.setPixmap(pixmap)
        self.scene = QGraphicsScene(self)
        self.scene.addItem(self.photo)

        text = self.scene.addText("Edited", f_top)
        text.setPos(offset, offset)
        text.setDefaultTextColor(text_color)
        text = self.scene.addText("(Toggle with {})".format(key), f_bottom)
        text.setDefaultTextColor(text_color)
        text.setPos(offset, offset + pixelsize + 10)

        # set up original scene
        self.scene_orig = QGraphicsScene(self)
        photo_orig = QGraphicsPixmapItem()
        photo_orig.setPixmap(pixmap_orig)
        self.scene_orig.addItem(photo_orig)

        text = self.scene_orig.addText("Original", f_top)
        text.setPos(offset, offset)
        text.setDefaultTextColor(text_color)
        text = self.scene_orig.addText("(Toggle with {})".format(key), f_bottom)
        text.setDefaultTextColor(text_color)
        text.setPos(offset, offset + pixelsize + 10)

        # create widget and set base scene
        self.setScene(self.scene)

    def fitInView(self, scale: bool = True):
        rect = QRectF(self.photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
            self.scale(1 / unity.width(), 1 / unity.height())
            viewrect = self.viewport().rect()
            scenerect = self.transform().mapRect(rect)
            factor = min(viewrect.width() / scenerect.width(),
                         viewrect.height() / scenerect.height())
            self.scale(factor, factor)
            self.zoom = 0
            # Center the image
            self.centerOn(rect.center())

    def toggleOriginal(self, original: bool):
        self.setScene(self.scene_orig if original else self.scene)


class AppearanceEditingController:
    def __init__(self, path_to_model, json_model, device, input_folder, output_folder, max_width=1024):
        # load pretrained bio model
        self.bio_skin_model = BioSkinInference(path_to_model + json_model, device=device)
        self.path_to_model = path_to_model
        self.json_model = json_model
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.device = device
        self.character_names = []
        self.characters = {}
        self.extensions = []
        self.edits_stack = []
        self.max_width = max_width

        self.character_names, self.extensions = io.get_file_list(input_folder)
        print("Found {} characters in the input directory. "
              "Estimating skin properties...".format(len(self.character_names)))

        for character_name, extension in zip(self.character_names, self.extensions):
            self.character = character_name
            print("Precomputing character {}".format(character_name))
            ch = Character(character_name, input_folder, device, extension)

            ch.load_input_albedo(max_width=max_width)
            ch.estimate_skin_props(self.bio_skin_model)
            ch.reconstruct_albedo(self.bio_skin_model, original=True)
            self.characters[character_name] = ch

        self.save_results_all(self.output_folder, save_skin_props=True, save_spectrum=True)
        print("Characters loaded!")
        check_memory(self.device)

    def setCharacter(self, character: str):
        self.character = character

    def resetAlbedo(self):
        self.characters[self.character].reset()

    def resetSkinProperty(self, i):
        self.characters[self.character].reset_param(SKIN_PROPS[i])

    def generateAlbedo(self):
        self.characters[self.character].reconstruct_albedo(self.bio_skin_model, original=False)

    def getAlbedo(self):
        return self.characters[self.character].albedo_map

    def getAlbedoEdited(self):
        return self.characters[self.character].albedo_map_edited

    def getSpecularMap(self):
        if self.characters[self.character].has_speculars:
            return self.characters[self.character].specular_map

    def getAlbedoIR(self):
        return self.characters[self.character].ir_map

    def getAlbedoIREdited(self):
        return self.characters[self.character].ir_map_edited

    def getComponent(self, str):
        return self.characters[self.character].get_param(str)

    def getComponentEdited(self, str):
        return self.characters[self.character].get_param_edited(str)

    def parameterManipulation(self, component: str, operator: int, sharpenValue: float,
                              increaseValue: float, offsetValue: float, flattenValue: float, blurrValue: float):
        if operator == 0 and increaseValue != 0:
            operation = "multiply" if increaseValue < 0 else "multiply"
            try:
                ch = self.characters[self.character]
                increaseValue = SKIN_MIN_VALUES[ch.param_index(component)] + \
                                increaseValue * SKIN_MAX_VALUES[ch.param_index(component)]
                ch.edit(component, operation, increaseValue)
            except Exception as ex:
                print(ex)
        if operator == 1 and offsetValue != 0:
            operation = "offset"
            try:
                ch = self.characters[self.character]
                ch.edit(component, operation, offsetValue)
            except Exception as ex:
                print(ex)
        if operator == 2 and flattenValue != 0:
            operation = "flatten_peaks" if flattenValue < 0 else "fill_valleys"
            try:
                ch = self.characters[self.character]
                parameter = ch.get_param(component)
                flattenValue = flattenValue * 2.576 * np.std(parameter) if flattenValue >= 0.0 \
                    else flattenValue * 1.4 * abs(np.max(parameter) - np.mean(parameter))
                ch.edit(component, operation, flattenValue)
            except Exception as ex:
                print(ex)

        if operator == 3 and sharpenValue != 0:
            operation = "smooth" if sharpenValue < 0 else "enhance"
            try:
                self.characters[self.character].edit(component, operation, abs(sharpenValue))
            except Exception as ex:
                print(ex)

        if operator == 4 and blurrValue != 0:
            operation = "blurr"
            try:
                ch = self.characters[self.character]
                ch.edit(component, operation, blurrValue)
            except Exception as ex:
                print(ex)

    def save_results(self, path, character_name, save_spectrum=False, save_skin_props=True):
        character_path = path + character_name + '/'
        print("Saving character " + character_path + "...")
        if not os.path.exists(character_path):
            os.mkdir(character_path)
        self.characters[character_name].save_reconstruction(self.bio_skin_model, character_path, save_spectrum,
                                                            save_skin_props, save_error=False)
        print("Done\n")

    def save_results_all(self, path, save_skin_props=True, save_spectrum=False):
        for index, character_name in enumerate(self.character_names):
            character_path = path + character_name + '/'
            print("Saving character " + character_path + "(" + str(index) + "/" +
                  str(len(self.character_names)) + ")...")
            if not os.path.exists(character_path):
                os.mkdir(character_path)
            self.characters[character_name].save_reconstruction(self.bio_skin_model, character_path, save_spectrum,
                                                                save_skin_props, save_error=False)
            print("Done\n")
        return 0

    def save_results_original_resolution(self, path, character_name, progress_bar=None, save_spectrum=False):
        character_path = path + character_name + '/'
        print("Saving character " + character_path + "...")
        if not os.path.exists(character_path):
            os.mkdir(character_path)
        reconstruct_in_batches_with_ops(device=self.device, path_to_model=self.path_to_model, json_model=None,
                                        bio_skin=self.bio_skin_model, input_folder=self.input_folder,
                                        output_folder=character_path, batch_size=1024, export_spectrums=save_spectrum,
                                        characters=self.characters, character_name=character_name,
                                        progress_bar=progress_bar)
        print("Done\n")
        return 0


class AppearanceEditing(QMainWindow):
    def __init__(self, path_to_model, json_model, device, input_folder, output_folder, max_width=1024, parent=None):
        super(AppearanceEditing, self).__init__(parent)
        # set up the appearance editing controller
        self.path_to_model = path_to_model
        self.json_model = json_model
        self.ae = AppearanceEditingController(path_to_model, json_model, device, input_folder, output_folder, max_width)
        self.latent_size = self.ae.bio_skin_model.latent_size
        # renderer path for export
        self.output_folder = output_folder
        self.max_width = max_width

        # basic window settings
        self.setWindowTitle("BioSkin")
        self.setWindowIcon(QIcon('ico.png'))
        menuBar = self.menuBar()

        # character menu for switching
        characterMenu = QMenu("&Character", self)
        for character in self.ae.character_names:
            action = QAction(character, characterMenu)
            action.triggered.connect(lambda _, c=character: self.setCharacter(c))
            characterMenu.addAction(action)
        menuBar.addMenu(characterMenu)

        # show options menu
        showMenu = QMenu("&View", self)
        self.toggleAction = QAction("&Original", showMenu, checkable=True)
        self.toggleAction.triggered.connect(
            lambda _: self.zoomWidget.toggleOriginal(self.toggleAction.isChecked()))
        showMenu.addAction(self.toggleAction)
        menuBar.addMenu(showMenu)

        self.toggleAction2 = QAction("&Diffuse Only", showMenu, checkable=True)
        self.toggleAction2.triggered.connect(self.updateZoomableImageShowButton)
        showMenu.addAction(self.toggleAction2)

        if json_model != 'BioSkinRGB':
            showAction1 = QAction("&Visible Range", showMenu)
            showAction1.triggered.connect(self.updateZoomableImageShowButton)
            showMenu.addAction(showAction1)

            showAction2 = QAction("&Near Infrared", showMenu)
            showAction2.triggered.connect(self.updateZoomableImageShowButtonIR)
            showMenu.addAction(showAction2)

        # Main save menu
        saveMenu = QMenu("&Save", self)
        # Save current submenu
        saveCurrentMenu = QMenu("&Save current", self)
        saveAction1 = QAction("&Save character", saveCurrentMenu)
        saveAction1.triggered.connect(lambda: self.ae.save_results(self.output_folder, self.ae.character))
        saveCurrentMenu.addAction(saveAction1)
        saveAction2 = QAction("&Save character and export spectrums", saveCurrentMenu)
        saveAction2.triggered.connect(lambda: self.ae.save_results(self.output_folder, self.ae.character,
                                                                   save_skin_props=True, save_spectrum=True))
        saveCurrentMenu.addAction(saveAction2)
        saveAction3 = QAction("&Save character original resolution", saveCurrentMenu)
        saveAction3.triggered.connect(lambda: self.ae.save_results_original_resolution(self.output_folder,
                                                                                       self.ae.character,
                                                                                       self.progressBar))
        saveCurrentMenu.addAction(saveAction3)
        # Add save current submenu to main save menu
        saveMenu.addMenu(saveCurrentMenu)
        # Save all submenu
        saveAllMenu = QMenu("&Save all", self)
        saveAction4 = QAction("&Save all", saveAllMenu)
        saveAction4.triggered.connect(lambda: self.ae.save_results_all(self.output_folder))
        saveAllMenu.addAction(saveAction4)
        saveAction5 = QAction("&Save all and export spectrums", saveAllMenu)
        saveAction5.triggered.connect(
            lambda: self.ae.save_results_all(self.output_folder, save_skin_props=True, save_spectrum=True))
        saveAllMenu.addAction(saveAction5)
        # Add save all submenu to main save menu
        saveMenu.addMenu(saveAllMenu)
        # Add main save menu to menu bar
        menuBar.addMenu(saveMenu)

        # create the progress bar
        self.createProgressBar()

        # set up zoomable view
        self.zoomWidget = ZoomView()
        self.setCentralWidget(self.zoomWidget)

        # add dock widget
        # Get the screen's geometry
        pixel_width = QDesktopWidget().screenGeometry().width()
        self.dockWidgetWidth = int(0.2 * pixel_width)
        self.rightWidget = QDockWidget()
        self.rightWidget.setFixedWidth(self.dockWidgetWidth)
        self.rightWidget.setFeatures(QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.RightDockWidgetArea, self.rightWidget)

        # set initial chracter
        self.setCharacter(self.ae.character_names[0])
        self.showMaximized()

        # initialize position
        self.zoomWidget.fitInView()

    def setCharacter(self, character: str):
        print("Setting character '{}'.".format(character))
        self.progressBar.setValue(33)
        self.ae.setCharacter(character)
        self.progressBar.setValue(66)
        self.zoomWidget.fitInView()
        self.updateDockWidget()
        albedo = self.ae.getAlbedo().copy()
        albedo_edited = self.ae.getAlbedoEdited().copy()

        specular_map = self.ae.getSpecularMap()
        if specular_map is not None:
            albedo += specular_map
            albedo_edited += specular_map

        self.zoomWidget.updateView(
            pixmap_from_array(albedo),
            pixmap_from_array(albedo_edited))
        self.progressBar.setValue(0)

    def keyPressEvent(self, event: QEvent):
        if event.key() == self.zoomWidget.toggleKey:
            self.toggleAction.setChecked(not self.toggleAction.isChecked())
            self.toggleAction.triggered.emit()
        elif event.key() == self.zoomWidget.toggleKey2:
            self.toggleAction2.setChecked(not self.toggleAction2.isChecked())
            self.toggleAction2.triggered.emit()
        event.accept()

    def eventFilter(self, source: QObject, event: QEvent):
        return super(AppearanceEditing, self).eventFilter(source, event)

    def save_for_renderer(self, image):
        start = time.time()
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)
        character = self.ae.character.replace('_albedo_baked', '')
        character = character + '/'
        character_dir = os.path.join(self.output_folder, character)
        if not os.path.exists(character_dir):
            os.mkdir(character_dir)
        textures_dir = os.path.join(character_dir, "textures/")
        if not os.path.exists(textures_dir):
            os.mkdir(textures_dir)
        image_to_renderer_path = os.path.join(textures_dir, "albedo_modified.exr")
        flag_renderer = os.path.join(textures_dir, "albedo_modified.flag")

        print("--> new reflectance saved to ", image_to_renderer_path)

        cv2.imwrite(image_to_renderer_path, image)

        if os.path.exists(flag_renderer):
            os.utime(flag_renderer, None)
        else:
            print("File does not exist: {}".format(flag_renderer))
        end = time.time()
        print("save_for_renderer takes {} seconds".format(end - start))

    def resetAlbedo(self):
        self.ae.resetAlbedo()
        self.setCharacter(self.ae.character)

    def resetSkinProperty(self, sliderEnhance, sliderIncrease, sliderOffset, sliderFlatten, sliderBlurr, label):
        self.progressBar.setValue(30)
        self.progressBar.setValue(60)
        self.ae.resetSkinProperty(self.currentActiveComponent)

        sliderEnhance.setValue(0)
        sliderIncrease.setValue(0)
        sliderOffset.setValue(0)
        sliderFlatten.setValue(0)
        sliderBlurr.setValue(0)

        # create pixmaps
        pixmap = pixmap_from_array(self.ae.getComponentEdited(SKIN_PROPS[self.currentActiveComponent]))
        pixmap_orig = pixmap_from_array(self.ae.getComponent(SKIN_PROPS[self.currentActiveComponent]))

        # update gui
        self.updateZoomableImage(label, pixmap, pixmap_orig)
        self.zoomWidget.updateView(pixmap, pixmap_orig)
        self.progressBar.setValue(0)

    def loadSkinProperty(self, label, param):
        ch = self.ae.characters[self.ae.character]
        index = ch.param_index(param)
        dir = self.output_folder + self.ae.character + '/' + self.ae.character + '_p' + str(index) + '_' + \
              SKIN_PROPS[index] + '.jpeg'
        map = io.load_image(path_to_image=dir, max_width=self.max_width)
        if map is not None:
            ch.parameters[index].update(image=map)
            # create pixmaps
            pixmap = pixmap_from_array(self.ae.getComponentEdited(param))
            pixmap_orig = pixmap_from_array(self.ae.getComponent(param))

            # update gui
            self.updateZoomableImage(label, pixmap, pixmap_orig)
            self.zoomWidget.updateView(pixmap, pixmap_orig)
            self.progressBar.setValue(0)
        else:
            print("ERROR: file not found in path:" + str(dir) + ". Save results first to be able upload either the "
                                                                "reconstructed or the edited (use the .jpeg) maps")

    def generateAlbedo(self):
        start = time.time()

        # run albedo generation
        self.progressBar.setValue(50)
        self.ae.generateAlbedo()

        albedo = self.ae.getAlbedo().copy()
        albedo_edited = self.ae.getAlbedoEdited().copy()

        specular_map = self.ae.getSpecularMap()
        if specular_map is not None and not self.toggleAction2.isChecked():
            albedo += specular_map
            albedo_edited += specular_map

        pixmap = pixmap_from_array(albedo_edited)
        # self.save_for_renderer(self.ae.getAlbedoEdited())

        # original
        pixmap_orig = pixmap_from_array(albedo)

        # set gui
        self.zoomWidget.updateView(pixmap, pixmap_orig)
        # self.updateZoomableImage(self.albedoLabel, pixmap, pixmap_orig)
        self.progressBar.setValue(0)

        end = time.time()
        print("Generate albedo takes {}".format(end - start))

    def updateZoomableImage(self, label: QLabel, pixmap: QPixmap, pixmap_orig: QPixmap):
        scaled = pixmap.scaledToWidth(self.dockWidgetWidth - 100)

        f = QFont()
        pixelsize = 14
        f.setPixelSize(pixelsize)
        f.setBold(True)
        offset = 20

        f_sub = QFont()
        f_sub.setPixelSize(pixelsize)

        # Begin painting on the QPixmap 'scaled'
        painter = QPainter(scaled)
        painter.setFont(f)

        # Set up path for main text to get drop shadow effect
        path = QPainterPath()
        path.addText(QPoint(offset, offset + pixelsize), f, "Miniview")

        # Create drop shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(5)
        shadow.setColor(QColor('black'))
        shadow.setOffset(1)

        # Draw the shadow by stroking the path
        painter.setPen(QPen(QColor('black'), 1, Qt.SolidLine))
        painter.setBrush(QColor('black'))
        painter.drawPath(path)

        # Overlay the text in white
        painter.setPen(QColor('white'))
        painter.fillPath(path, QColor('white'))

        # Draw subtext normally, since no shadow effect is specified
        painter.setFont(f_sub)
        painter.setPen(QColor('white'))
        painter.drawText(QPoint(offset, offset + 2 * pixelsize + 5), "(Click to enlarge)")

        painter.end()

        label.setPixmap(scaled)
        label.mousePressEvent = \
            lambda event, pixmap=pixmap, pixmap_orig=pixmap_orig: self.zoomWidget.updateView(pixmap, pixmap_orig)

    # update images after parameter manipulation
    def parameterManipulation(self, sliderEnhance, sliderIncrease, sliderOffset, sliderFlatten, sliderBlurr, label,
                              component, operation):
        start = time.time()
        self.progressBar.setValue(30)

        # edit
        sharpenValue = sliderEnhance.value() / 100
        increaseValue = sliderIncrease.value() / 100
        offsetValue = sliderOffset.value() / 100
        flattenValue = sliderFlatten.value() / 100
        blurrValue = sliderBlurr.value() / 100
        self.ae.parameterManipulation(component, operation, sharpenValue, increaseValue, offsetValue, flattenValue,
                                      blurrValue)
        self.progressBar.setValue(60)

        # create pixmaps
        pixmap = pixmap_from_array(self.ae.getComponentEdited(component))
        pixmap_orig = pixmap_from_array(self.ae.getComponent(component))

        # update gui
        self.updateZoomableImage(label, pixmap, pixmap_orig)
        self.zoomWidget.updateView(pixmap, pixmap_orig)
        self.progressBar.setValue(0)

        end = time.time()
        # print("Manipulate parameters takes {}".format(end - start))

    def updateZoomableImageShowButton(self):
        albedo = self.ae.getAlbedo().copy()
        albedo_edited = self.ae.getAlbedoEdited().copy()

        specular_map = self.ae.getSpecularMap()
        if specular_map is not None and not self.toggleAction2.isChecked():
            albedo += specular_map
            albedo_edited += specular_map

        pixmap = pixmap_from_array(albedo_edited)
        pixmap_orig = pixmap_from_array(albedo)
        self.zoomWidget.updateView(pixmap, pixmap_orig)

    def updateZoomableImageShowButtonIR(self):
        pixmap = pixmap_from_array(self.ae.getAlbedoIREdited())
        pixmap_orig = pixmap_from_array(self.ae.getAlbedoIR())
        self.zoomWidget.updateView(pixmap, pixmap_orig)

    def updateDockWidget(self):
        layout = QVBoxLayout()
        button_update = QPushButton("Push Edits and Update Reflectance")
        button_update.clicked.connect(self.generateAlbedo)
        button_reset = QPushButton("Reset Reflectance")
        button_reset.clicked.connect(self.resetAlbedo)
        boxlayout = QVBoxLayout()
        boxlayout.addWidget(button_update)
        boxlayout.addWidget(button_reset)
        groupbox = QGroupBox("Reflectance")
        groupbox.setLayout(boxlayout)
        layout.addWidget(groupbox)

        # add component parameter widgets
        count = 0
        columns = 2
        componentWidget = QTabWidget()
        componentWidget.currentChanged.connect(self.updateCurrentActiveComponent)
        ch = self.ae.characters[self.ae.character]

        for i in range(0, self.latent_size):
            component = SKIN_PROPS[i]
            newWidget = QWidget()

            labelEnhance = QLabel("Smooth/Enhance:", newWidget)
            labelEnhance.setToolTip("Smooth/Enhance the distribution")
            sliderEnhance = QSlider(Qt.Horizontal, newWidget)
            sliderEnhance.setRange(-100, 100)
            sliderEnhance.setValue(0)
            valueSharpen = QLabel(newWidget)
            valueSharpen.setFixedWidth(40)

            labelBlurr = QLabel("Blurr:", newWidget)
            labelBlurr.setToolTip("Gaussian Blurr")
            sliderBlurr = QSlider(Qt.Horizontal, newWidget)
            sliderBlurr.setRange(0, 100)
            sliderBlurr.setValue(0)
            valueBlurr = QLabel(newWidget)
            valueBlurr.setFixedWidth(40)

            labelIncrease = QLabel("Scale:", newWidget)
            labelIncrease.setToolTip("Scales the pixel values by a multiplier.")
            sliderIncrease = QSlider(Qt.Horizontal, newWidget)
            sliderIncrease.setRange(-100, 100)
            sliderIncrease.setValue(0)
            valueIncrease = QLabel(newWidget)

            labelOffset = QLabel("Offset:", newWidget)
            labelOffset.setToolTip("Adds or subtracts a constant value to every pixel.")
            sliderOffset = QSlider(Qt.Horizontal, newWidget)
            sliderOffset.setRange(-100, 100)
            sliderOffset.setValue(0)
            valueOffset = QLabel(newWidget)

            labelFlatten = QLabel("Floor/Ceiling:", newWidget)
            labelFlatten.setToolTip("Adds value to pixels below a certain threshold / "
                                    "value is subtracted from pixels above a certain threshold.")
            sliderFlatten = QSlider(Qt.Horizontal, newWidget)
            sliderFlatten.setRange(-100, 100)
            sliderFlatten.setValue(0)
            valueFlatten = QLabel(newWidget)

            labelComponent = QLabel()


            # add scale operator to the gui
            sliderIncrease.valueChanged.connect(lambda value, label=valueIncrease: label.setText(str(value / 100)))
            sliderIncrease.valueChanged.connect(
                lambda _, sliderEnhance=sliderEnhance, sliderIncrease=sliderIncrease, sliderOffset=sliderOffset,
                       sliderFlatten=sliderFlatten, sliderBlurr=sliderBlurr, label=labelComponent, component=component:
                self.parameterManipulation(sliderEnhance, sliderIncrease, sliderOffset, sliderFlatten, sliderBlurr,
                                           label, component, 0))
            sliderIncrease.valueChanged.emit(sliderIncrease.value())

            # add offset operator to the gui
            sliderOffset.valueChanged.connect(lambda value, label=valueOffset: label.setText(str(value / 100)))
            sliderOffset.valueChanged.connect(
                lambda _, sliderEnhance=sliderEnhance, sliderIncrease=sliderIncrease, sliderOffset=sliderOffset,
                       sliderFlatten=sliderFlatten, sliderBlurr=sliderBlurr, label=labelComponent, component=component:
                self.parameterManipulation(sliderEnhance, sliderIncrease, sliderOffset, sliderFlatten, sliderBlurr,
                                           label, component, 1))
            sliderOffset.valueChanged.emit(sliderOffset.value())

            # add flatten to the gui
            sliderFlatten.valueChanged.connect(lambda value, label=valueFlatten: label.setText(str(value / 100)))
            sliderFlatten.valueChanged.connect(
                lambda _, sliderEnhance=sliderEnhance, sliderIncrease=sliderIncrease, sliderOffset=sliderOffset,
                       sliderFlatten=sliderFlatten, sliderBlurr=sliderBlurr, label=labelComponent, component=component:
                self.parameterManipulation(sliderEnhance, sliderIncrease, sliderOffset, sliderFlatten, sliderBlurr,
                                           label, component, 2))
            sliderFlatten.valueChanged.emit(sliderFlatten.value())

            # add smooth/sharpen operator to the gui
            sliderEnhance.valueChanged.connect(lambda value, label=valueSharpen: label.setText(str(value / 100)))
            sliderEnhance.valueChanged.connect(lambda value, sliderIncrease=sliderIncrease: sliderIncrease.setValue(0))
            sliderEnhance.valueChanged.connect(
                lambda _, sliderEnhance=sliderEnhance, sliderIncrease=sliderIncrease, sliderOffset=sliderOffset,
                       sliderFlatten=sliderFlatten, sliderBlurr=sliderBlurr, label=labelComponent, component=component:
                self.parameterManipulation(sliderEnhance, sliderIncrease, sliderOffset, sliderFlatten, sliderBlurr,
                                           label, component, 3))
            sliderEnhance.valueChanged.emit(sliderEnhance.value())

            # add blurr to the gui
            sliderBlurr.valueChanged.connect(lambda value, label=valueBlurr: label.setText(str(value / 100)))
            sliderBlurr.valueChanged.connect(
                lambda _, sliderEnhance=sliderEnhance, sliderIncrease=sliderIncrease, sliderOffset=sliderOffset,
                       sliderFlatten=sliderFlatten, sliderBlurr=sliderBlurr, label=labelComponent, component=component:
                self.parameterManipulation(sliderEnhance, sliderIncrease, sliderOffset, sliderFlatten, sliderBlurr,
                                           label, component, 4))
            sliderFlatten.valueChanged.emit(sliderFlatten.value())

            boxlayout = QGridLayout()

            # scale
            boxlayout.addWidget(labelIncrease, 0, 0)
            boxlayout.addWidget(sliderIncrease, 0, 1)
            boxlayout.addWidget(valueIncrease, 0, 2)

            # offset
            boxlayout.addWidget(labelOffset, 1, 0)
            boxlayout.addWidget(sliderOffset, 1, 1)
            boxlayout.addWidget(valueOffset, 1, 2)

            # smooth/sharpen
            boxlayout.addWidget(labelEnhance, 2, 0)
            boxlayout.addWidget(sliderEnhance, 2, 1)
            boxlayout.addWidget(valueSharpen, 2, 2)

            # flatten
            boxlayout.addWidget(labelFlatten, 3, 0)
            boxlayout.addWidget(sliderFlatten, 3, 1)
            boxlayout.addWidget(valueFlatten, 3, 2)

            # # blurr
            # boxlayout.addWidget(labelBlurr, 3, 0)
            # boxlayout.addWidget(sliderBlurr, 3, 1)
            # boxlayout.addWidget(valueBlurr, 3, 2)

            button_reset_parameter = QPushButton("Reset Property")
            button_reset_parameter.clicked.connect(lambda _, sliderEnhance=sliderEnhance, sliderIncrease=sliderIncrease,
                                                          sliderOffset=sliderOffset, sliderFlatten=sliderFlatten,
                                                          sliderBlurr=sliderBlurr, label=labelComponent:
                                                   self.resetSkinProperty(sliderEnhance, sliderIncrease, sliderOffset,
                                                                          sliderFlatten, sliderBlurr, label))
            boxlayout.addWidget(button_reset_parameter, 4, 1)

            button_load_parameter = QPushButton("Load from File")
            button_load_parameter.clicked.connect(lambda _, label=labelComponent, param=component:
                                                  self.loadSkinProperty(label, param))
            boxlayout.addWidget(button_load_parameter, 5, 1)

            # images
            groupbox = QTabWidget()
            groupbox.addTab(labelComponent, component)
            boxlayout.addWidget(groupbox, 6, 0, 1, 6)

            newWidget.setLayout(boxlayout)
            componentWidget.addTab(newWidget, component)
            count += 1

        # create main content widget
        layout.addWidget(componentWidget)
        widget = QWidget()
        widget.setLayout(layout)

        self.rightWidget.setWidget(widget)
        self.rightWidget.setWidget(widget)

    def updateCurrentActiveComponent(self, index):
        self.currentActiveComponent = index

    def createProgressBar(self):
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)

        self.lower = QStatusBar()
        self.lower.addPermanentWidget(self.progressBar, 1)
        self.setStatusBar(self.lower)


def check_memory(device):
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(device)
        total_memory_gb = gpu_properties.total_memory / (1024 ** 3)
        print(f"Total GPU Memory: {total_memory_gb:.2f} GB")

        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        allocated_mb = allocated / (1024 ** 3)
        cached_mb = cached / (1024 ** 3)
        print(f"Allocated memory: {allocated_mb:.2f} GB")
        print(f"Cached memory: {cached_mb:.2f} GB")


def run_gui(args):

    print(torch.__version__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device) + "(" + str(torch.cuda.device_count()) + ")")
    check_memory(device)

    # Qt
    app = QApplication([])
    # primary_screen = app.primaryScreen()
    # dpi = primary_screen.physicalDotsPerInch()
    # pixelsize = 20
    # pixelsize = int(pixelsize * (dpi / 96.0))  # 96 DPI is a common standard

    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ae = AppearanceEditing(args.path_to_model, args.json_model, device, args.input_folder, args.output_folder,
                           args.max_width)
    ae.show()

    # Loop
    sys.exit(app.exec())


if __name__ == '__main__':
    args = parse_arguments()
    run_gui()
