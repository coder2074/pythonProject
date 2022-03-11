import language_tool_python
tool = language_tool_python.LanguageTool('en-US')
#bad_text = 'A sentence with a error in the Hitchhiker’s Guide tot he Galaxy'
bad_text = 'cut way much'
matches = tool.check(bad_text)
len(matches)
print(matches)
corrected_text = tool.correct(bad_text)
tool.close() # Call `close()` to shut off the server when you're done.

print('bad text')
print(bad_text)

print('corrected text')
print(corrected_text)

print('done')
