#
# Makefile
#
#
#

###########################

# You need to edit these values.

DICT_NAME		=	"新明解IT辞典"
DICT_SRC_PATH		=	MyDictionary.xml
CSS_PATH		=	MyDictionary.css
PLIST_PATH		=	MyInfo.plist

DICT_BUILD_OPTS		=
# Suppress adding supplementary key.
# DICT_BUILD_OPTS		=	-s 0	# Suppress adding supplementary key.

###########################

# The DICT_BUILD_TOOL_DIR value is used also in "build_dict.sh" script.
# You need to set it when you invoke the script directly.

DICT_BUILD_TOOL_DIR	=	"${HOME}/DevTools/Utilities/Dictionary Development Kit"
DICT_BUILD_TOOL_BIN	=	"$(DICT_BUILD_TOOL_DIR)/bin"

###########################

DICT_DEV_KIT_OBJ_DIR	=	./objects
export	DICT_DEV_KIT_OBJ_DIR

DESTINATION_FOLDER	=	~/Library/Dictionaries
RM			=	/bin/rm

###########################

JING_URL	=	https://repo1.maven.org/maven2/com/thaiopensource/jing/20091111/jing-20091111.jar
JING_JAR	=	jing-20091111.jar

###########################

all:
	"$(DICT_BUILD_TOOL_BIN)/build_dict.sh" $(DICT_BUILD_OPTS) $(DICT_NAME) $(DICT_SRC_PATH) $(CSS_PATH) $(PLIST_PATH)
	echo "Done."


install:
	echo "Installing into $(DESTINATION_FOLDER)".
	mkdir -p $(DESTINATION_FOLDER)
	ditto --noextattr --norsrc $(DICT_DEV_KIT_OBJ_DIR)/$(DICT_NAME).dictionary  $(DESTINATION_FOLDER)/$(DICT_NAME).dictionary
	touch $(DESTINATION_FOLDER)
	echo "Done."
	echo "To test the new dictionary, try Dictionary.app."

$(JING_JAR):
	wget $(JING_URL) -O $(JING_JAR)

validate: $(JING_JAR);
	java -jar $(JING_JAR) documents/DictionarySchema/AppleDictionarySchema.rng $(DICT_SRC_PATH)

clean:
	$(RM) -rf $(DICT_DEV_KIT_OBJ_DIR)
