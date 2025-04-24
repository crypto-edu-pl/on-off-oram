ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)

ifeq ($(DEBUG),1)
GRAMINE_LOG_LEVEL = debug
else
GRAMINE_LOG_LEVEL = error
endif

TESTS = binsearch_simple spellcheck

all: $(TESTS)

.PHONY: $(TESTS)
$(TESTS): %: bin/% %.manifest

ifeq ($(SGX), 1)
$(TESTS): %: %.manifest.sgx %.sig
endif

$(addsuffix .manifest,$(TESTS)): %: %.template
	gramine-manifest -Dlog_level=$(GRAMINE_LOG_LEVEL) -Darch_libdir=$(ARCH_LIBDIR) $< > $@

define sgx_generation_template
$(1).sig $(1).manifest.sgx &: $(1).manifest
	gramine-sgx-sign --manifest $$< --output $(1).manifest.sgx
endef
$(foreach test,$(TESTS),$(eval $(call sgx_generation_template,$(test))))

.PHONY: $(addprefix bin/,$(TESTS))
$(addprefix bin/,$(TESTS)): bin/%:
	cargo install --features ""$(FEATURES) --path . --root . --bin $*

.PHONY: clean
clean:
	$(RM) *.token *.sig *.manifest.sgx *.manifest
	cargo clean
	rm -rf bin/
